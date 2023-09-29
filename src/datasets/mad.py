import os
import h5py
import random
import math
import numpy as np
from easydict import EasyDict as edict
from collections import defaultdict
import torch
import torch.utils.data as data

from ..utils import compute_overlap


class MADDataset(data.Dataset):
    
    def __init__(self, split, cfg, pre_load=False):
        super().__init__()

        self.split = split
        self.data_dir = cfg.DATA.DATA_DIR
        self.snippet_length = cfg.MODEL.SNIPPET_LENGTH
        self.scale_num = cfg.MODEL.SCALE_NUM
        self.max_anchor_length = self.snippet_length * 2**(self.scale_num - 1)
        if split == "train":
            epochs = cfg.TRAIN.NUM_EPOCH
            batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            epochs = 1
            batch_size = 1000000
        
        self.q2v = dict()
        self.v2q = defaultdict(list)
        self.v2dur = dict()
        with open(os.path.join(self.data_dir, f"annotations/{split}.txt"), 'r') as f:
            for i, line in enumerate(f.readlines()):
                qid, vid, duration, start, end, text = line.strip().split(" | ")
                qid = int(qid)

                assert float(start) < float(end), \
                    "Wrong timestamps for {}: start >= end".format(qid)
                
                if vid not in self.v2dur:
                    self.v2dur[vid] = float(duration)
                self.q2v[qid] = {
                    "vid": vid,
                    "duration": float(duration),
                    "timestamps": [float(start), float(end)],
                    "text": text.lower()
                }
                self.v2q[vid].append(qid)
        
        # generate training batch
        self.samples = list()
        for i_epoch in range(epochs):
            batches = list()
            for vid, qids in self.v2q.items():
                cqids = qids.copy()
                if self.split == "train":
                    random.shuffle(cqids)
                    if len(cqids) % batch_size != 0:
                        pad_num = batch_size - len(cqids) % batch_size
                        cqids = cqids + cqids[:pad_num]
                
                steps = np.math.ceil(len(cqids) / batch_size)
                for j in range(steps):
                    batches.append({"vid": vid, "qids": cqids[j*batch_size:(j+1)*batch_size]})
            
            if self.split == "train":
                random.shuffle(batches)
            self.samples.extend(batches)

        self.vfeat_path = os.path.join(self.data_dir, "features/CLIP_frames_features_5fps.h5")
        self.qfeat_path = os.path.join(self.data_dir, "features/CLIP_language_sentence_features.h5")
        if pre_load:
            with h5py.File(self.vfeat_path, 'r') as f:
                self.vfeats = {m: np.asarray(f[m]) for m in self.v2q.keys()}
            with h5py.File(self.qfeat_path, 'r') as f:
                self.qfeats = {str(m): np.asarray(f[str(m)]) for m in self.q2v.keys()}
        else:
            self.vfeats, self.qfeats = None, None
        self.fps = 5.0


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        vid = self.samples[idx]["vid"]
        qids = self.samples[idx]["qids"]
        duration = self.v2dur[vid]

        if not self.vfeats:
            self.vfeats = h5py.File(self.vfeat_path, 'r')
        ori_video_feat = np.asarray(self.vfeats[vid])
        ori_video_length, feat_dim = ori_video_feat.shape
        pad_video_length = int(np.math.ceil(ori_video_length / self.max_anchor_length) * self.max_anchor_length)
        pad_video_feat = np.zeros((pad_video_length, feat_dim), dtype=float)
        pad_video_feat[:ori_video_length, :] = ori_video_feat
        
        querys = {
            "texts": list(),
            "query_feats": list(),
            "query_masks": list(),
            "anchor_masks": list(),
            "starts": list(),
            "ends": list(),
            "overlaps": list(),
            "timestamps": list(),
        }
        scale_boundaries = [0]
        for qid in qids:
            text = self.q2v[qid]["text"]
            timestamps = self.q2v[qid]["timestamps"]
            if not self.qfeats:
                self.qfeats = h5py.File(self.qfeat_path, 'r')
            query_feat = np.asarray(self.qfeats[str(qid)])
            query_length = query_feat.shape[0]
            query_mask = np.ones((query_length, ), dtype=float)

            # generate multi-level groundtruth
            masks, starts, ends, overlaps = list(), list(), list(), list()
            for i in range(self.scale_num):
                anchor_length = self.snippet_length * 2**i
                nfeats = math.ceil(ori_video_length / anchor_length)
                s_times = np.arange(0, nfeats).astype(np.float32) * (anchor_length / self.fps)
                e_times = np.minimum(duration, np.arange(1, nfeats + 1).astype(np.float32) * (anchor_length / self.fps))
                candidates = np.stack([s_times, e_times], axis=1)
                overlap = compute_overlap(candidates.tolist(), timestamps)
                mask = np.ones((nfeats, ), dtype=int)

                pad_nfeats = math.ceil(pad_video_length / anchor_length)
                starts.append(self.pad(s_times, pad_nfeats))
                ends.append(self.pad(e_times, pad_nfeats))
                overlaps.append(self.pad(overlap, pad_nfeats))
                masks.append(self.pad(mask, pad_nfeats))

                if len(scale_boundaries) != self.scale_num + 1:
                    scale_boundaries.append(scale_boundaries[-1] + pad_nfeats)

            starts = np.concatenate(starts, axis=0)
            ends = np.concatenate(ends, axis=0)
            overlaps = np.concatenate(overlaps, axis=0)
            masks = np.concatenate(masks, axis=0)

            querys["texts"].append(text)
            querys["query_feats"].append(torch.from_numpy(query_feat))
            querys["query_masks"].append(torch.from_numpy(query_mask))
            querys["anchor_masks"].append(torch.from_numpy(masks))
            querys["starts"].append(torch.from_numpy(starts))
            querys["ends"].append(torch.from_numpy(ends))
            querys["overlaps"].append(torch.from_numpy(overlaps))
            querys["timestamps"].append(torch.FloatTensor(timestamps))

        instance = {
            "vid": vid,
            "duration": float(duration),
            "video_feats": torch.from_numpy(pad_video_feat).unsqueeze(0).float(),
            "scale_boundaries": torch.LongTensor(scale_boundaries),
            "qids": qids,
            "texts":querys["texts"],
            "query_feats": torch.stack(querys["query_feats"], dim=0).float(),
            "query_masks": torch.stack(querys["query_masks"], dim=0).float(),
            "anchor_masks": torch.stack(querys["anchor_masks"], dim=0),
            "starts":  torch.stack(querys["starts"], dim=0),
            "ends": torch.stack(querys["ends"], dim=0),
            "overlaps": torch.stack(querys["overlaps"], dim=0),
            "timestamps": torch.stack(querys["timestamps"], dim=0)
        }
        return instance 


    def pad(self, arr, pad_len):
        new_arr = np.zeros((pad_len, ), dtype=float)
        new_arr[:len(arr)] = arr
        return new_arr


    @staticmethod
    def collate_fn(data):
        all_items = data[0].keys()
        no_tensor_items = ["vid", "duration", "qids", "texts"]

        batch = {k: [d[k] for d in data] for k in all_items}
        for k in all_items:
            if k not in no_tensor_items:
                batch[k] = torch.cat(batch[k], dim=0)
        
        return batch



if __name__ == "__main__":
    import yaml
    with open("conf/soonet_mad.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
        print(cfg)

    mad_dataset = MADDataset("train", cfg)    
    data_loader = data.DataLoader(mad_dataset, 
                            batch_size=1,
                            num_workers=4,
                            shuffle=False,
                            collate_fn=mad_dataset.collate_fn,
                            drop_last=False
                        )

    for i, batch in enumerate(data_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print("{}: {}".format(k, v.size()))
            else:
                print("{}: {}".format(k, v))
        break