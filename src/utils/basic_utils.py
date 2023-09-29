import numpy as np
import random
import logging, logging.handlers
import coloredlogs
import torch


def get_logger(name, log_file_path=None, fmt="%(asctime)s %(name)s: %(message)s",
               print_lev=logging.DEBUG, write_lev=logging.INFO):
    logger = logging.getLogger(name)
    # Add file handler
    if log_file_path:
        formatter = logging.Formatter(fmt)
        file_handler = logging.handlers.RotatingFileHandler(log_file_path)
        file_handler.setLevel(write_lev)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # Add stream handler
    coloredlogs.install(level=print_lev, logger=logger,
                        fmt="%(asctime)s %(name)s %(message)s")
    return logger


def count_parameters(model):
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        train_params += parameter.numel()
    print(f"Total Trainable Params: {train_params}")



def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def compute_tiou(pred, gt):
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return float(intersection) / (union + 1e-9)


def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def fetch_feats_by_index(ori_feats, indices):
    B, L = indices.shape
    filtered_feats = ori_feats[torch.arange(B)[:, None], indices]
    return filtered_feats



class Evaluator(object):

    def __init__(self, tiou_threshold=[0.1, 0.3, 0.5], topks=[1, 5, 10, 50, 100]):
        self.tiou_threshold = tiou_threshold
        self.topks = topks

    def eval_instance(self, pred, gt, topk):
        """ Compute Recall@topk at predefined tiou threshold for instance
        Args:
            pred: predictions of starting/end position; list of [start,end]
            gt: ground-truth of starting/end position; [start,end]
            topk: rank of predictions; int
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        correct = {str(tiou):0 for tiou in self.tiou_threshold}
        find = {str(tiou):False for tiou in self.tiou_threshold}
        if len(pred) == 0:
            return correct

        if len(pred) > topk:
            pred = pred[:topk]

        best_tiou = 0
        for loc in pred:
            cur_tiou = compute_tiou(loc, gt)

            if cur_tiou > best_tiou:
                best_tiou = cur_tiou

            for tiou in self.tiou_threshold:
                if (not find[str(tiou)]) and (cur_tiou >= tiou):
                    correct[str(tiou)] = 1
                    find[str(tiou)] = True

        return correct, best_tiou

    def eval(self, preds, gts):
        """ Compute R@1 and R@5 at predefined tiou threshold [0.3,0.5,0.7]
        Args:
            pred: predictions consisting of starting/end position; list
            gt: ground-truth of starting/end position; [start,end]
        Return:
            correct: flag of correct at predefined tiou threshold [0.3,0.5,0.7]
        """
        num_instances = float(len(preds))
        miou = 0
        all_rank = dict()
        for tiou in self.tiou_threshold:
            for topk in self.topks:
                all_rank["R{}-{}".format(topk, tiou)] = 0

        for pred,gt in zip(preds, gts):
            for topk in self.topks:
                correct, iou = self.eval_instance(pred, gt, topk=topk)
                for tiou in self.tiou_threshold:
                    all_rank["R{}-{}".format(topk, tiou)] += correct[str(tiou)]

                # miou += iou

        for tiou in self.tiou_threshold:
            for topk in self.topks:
                all_rank["R{}-{}".format(topk, tiou)] /= num_instances

        # miou /= float(num_instances)
        
        return all_rank, miou