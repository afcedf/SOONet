import os
import json
import time
from tqdm import tqdm
import torch
from torch.optim import AdamW, lr_scheduler

from .models import SOONet
from .utils import Evaluator, get_logger


class Trainer(object):

    def __init__(self, mode, save_or_load_path, cfg):
        self.device = torch.device(cfg.device_id) if torch.cuda.is_available() else torch.device("cpu")
        self.model = SOONet(cfg)
        
        self.evaluator = Evaluator(tiou_threshold=cfg.TEST.EVAL_TIOUS, topks=cfg.TEST.EVAL_TOPKS)

        self.save_or_load_path = save_or_load_path
        log_dir = os.path.join(save_or_load_path, "log")
        ckpt_dir = os.path.join(save_or_load_path, "ckpt")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.cfg = cfg

        if mode == "train":
            with open(os.path.join(log_dir, "config.json"), 'w') as f:
                js = json.dumps(cfg, indent=2)
                f.write(js)

            self.optimizer = self.build_optimizer(cfg)
            self.scheduler = lr_scheduler.StepLR(self.optimizer, cfg.OPTIMIZER.LR_DECAY_STEP, 
                                    gamma=cfg.OPTIMIZER.LR_DECAY, last_epoch=-1, verbose=False)
        
    
    def build_optimizer(self, cfg):
        no_decay = ['bias', 'layer_norm', 'LayerNorm']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.OPTIMIZER.WD},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.OPTIMIZER.LR)
        
        return optimizer


    def train(self, train_loader, test_loader):
        logger = get_logger("TRAIN", log_file_path=os.path.join(self.log_dir, "train.log"))
        self.model.to(self.device)
        self.train_epoch(0, train_loader, test_loader, logger, self.cfg)


    def eval(self, test_loader):
        logger = get_logger("EVAL", log_file_path=os.path.join(self.log_dir, "eval.log"))

        resume_path = os.path.join(self.ckpt_dir, "best.pth")
        logger.info("Load trained model from: {}".format(resume_path))

        saver_dict = torch.load(resume_path, map_location="cpu")
        state_dict = saver_dict["model"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Load trained model succeed.")

        all_rank, miou = self.eval_epoch(test_loader)
        for k, v in all_rank.items():
            logger.info("{}: {:.4f}".format(k, v))


    def test(self, test_loader):
        logger = get_logger("TEST", log_file_path=os.path.join(self.log_dir, "test.log"))

        resume_path = os.path.join(self.ckpt_dir, "best.pth")
        logger.info("Load trained model from: {}".format(resume_path))

        saver_dict = torch.load(resume_path, map_location="cpu")
        state_dict = saver_dict["model"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Load trained model succeed.")

        start = time.time()
        test_instances = []
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                scores, bboxes = self.model(
                    query_feats=batch["query_feats"].to(self.device), 
                    query_masks=batch["query_masks"].to(self.device), 
                    video_feats=batch["video_feats"].to(self.device),
                    start_ts=batch["starts"].to(self.device),
                    end_ts=batch["ends"].to(self.device),
                    scale_boundaries=batch["scale_boundaries"].to(self.device),
                )
                for i in range(len(bboxes)):
                    instance = {
                        "vid": batch["vid"],
                        "duration": batch["duration"],
                        "qid": batch["qids"][0][i],
                        "text": batch["texts"][0][i],
                        "timestamp": batch["timestamps"][i].numpy().tolist(), 
                        "pred_scores": scores[i],
                        "pred_bboxes": bboxes[i]
                    }
                    test_instances.append(instance)

        logger.info("cost time: {}".format(time.time() - start))
        result_path = os.path.join(self.log_dir, "infer_result.json")
        with open(result_path, 'w') as f:
            res = json.dumps(test_instances, indent=2)
            f.write(res)


    def train_epoch(self, epoch, train_loader, test_loader, logger, cfg):
        self.model.train()

        best_r1 = 0
        for i, batch in enumerate(train_loader):
            loss_dict = self.model(
                query_feats=batch["query_feats"].to(self.device), 
                query_masks=batch["query_masks"].to(self.device), 
                video_feats=batch["video_feats"].to(self.device),
                start_ts=batch["starts"].to(self.device),
                end_ts=batch["ends"].to(self.device),
                scale_boundaries=batch["scale_boundaries"].to(self.device),
                overlaps=batch["overlaps"].to(self.device),
                timestamps=batch["timestamps"].to(self.device),
                anchor_masks=batch["anchor_masks"].to(self.device)
            )
            total_loss = loss_dict["total_loss"]
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if i % cfg.TRAIN.LOG_STEP == 0:
                log_str = f"Step: {i}, "
                for k, v in loss_dict.items():
                    log_str += "{}: {:.3f}, ".format(k, v)
                logger.info(log_str[:-2])
            
            if i > 0 and i % cfg.TRAIN.EVAL_STEP == 0:
                all_rank, miou = self.eval_epoch(test_loader)

                logger.info("step: {}".format(i))
                for k, v in all_rank.items():
                    logger.info("{}: {:.4f}".format(k, v))

                r1 = all_rank["R1-0.5"]
                if r1 > best_r1:
                    best_r1 =r1
                    saver_dict = {
                        "step": i,
                        "r1-0.5": r1,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()
                    }
                    save_path = os.path.join(self.ckpt_dir, "best.pth")
                    torch.save(saver_dict, save_path)
                
                self.model.train()
        
        logger.info("best R1-0.5: {:.4f}".format(best_r1))



    def eval_epoch(self, test_loader):
        self.model.eval()

        preds, gts = list(), list()
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                scores, bboxes = self.model(
                    query_feats=batch["query_feats"].to(self.device), 
                    query_masks=batch["query_masks"].to(self.device), 
                    video_feats=batch["video_feats"].to(self.device),
                    start_ts=batch["starts"].to(self.device),
                    end_ts=batch["ends"].to(self.device),
                    scale_boundaries=batch["scale_boundaries"].to(self.device),
                )
                preds.extend(bboxes)
                gts.extend([i for i in batch["timestamps"].numpy()])

        return self.evaluator.eval(preds, gts)