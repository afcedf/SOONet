import argparse
import yaml
from easydict import EasyDict as edict
import torch
import torch.utils.data as data

from .datasets import *
from .trainer import Trainer
from .utils import set_seed



def main():
    parser = argparse.ArgumentParser("Setting for training SOONet Models")

    parser.add_argument("--exp_path", type=str)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")

    opt = parser.parse_args()

    config_path = "conf/{}.yaml".format(opt.config_name)
    with open(config_path, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    cfg.device_id = opt.device_id
    torch.cuda.set_device(opt.device_id)
    set_seed(cfg.SEED)
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    dset = cfg.DATASET
    if dset.lower() == "mad":
        trainset = MADDataset("train", cfg, pre_load=cfg.DATA.PRE_LOAD) if opt.mode == "train" else list()
        testset = MADDataset("test", cfg, pre_load=cfg.DATA.PRE_LOAD)
    else:
        raise NotImplementedError
    
    print("Train batch num: {}, Test batch num: {}".format(len(trainset), len(testset)))
    print(cfg)

    if opt.mode == "train":
        train_loader = data.DataLoader(trainset, 
                                batch_size=1,
                                num_workers=cfg.TRAIN.WORKERS,
                                shuffle=False,
                                collate_fn=trainset.collate_fn,
                                drop_last=False
                            )

    test_loader = data.DataLoader(testset, 
                            batch_size=1,
                            num_workers=cfg.TEST.WORKERS,
                            shuffle=False,
                            collate_fn=testset.collate_fn,
                            drop_last=False
                        )

    trainer = Trainer(mode=opt.mode, save_or_load_path=opt.exp_path, cfg=cfg)
    
    if opt.mode == "train":
        trainer.train(train_loader, test_loader)
    elif opt.mode == "eval":
        trainer.eval(test_loader)
    elif opt.mode == "test":
        trainer.test(test_loader)
    else:
        raise ValueError(f'The value of mode {opt.mode} is not in ["train", "eval", "test"]')



if __name__ == "__main__":
    main()