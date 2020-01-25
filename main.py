import argparse
import torch
import mmcv
import random
import numpy as np
import torch.optim as optim
from model.ITAE import Model
from data_tools.data import load_data
from torch.utils.data import DataLoader
from train import train_epoch, test_epoch
from mmcv import Config
from data_tools.utils import log_txt, L1_measure


def parse_args():
    parse = argparse.ArgumentParser(description='lip_language Model')
    parse.add_argument('config_file', help='the path of config file')
    parse.add_argument('--work_dir', help='the dir to save logs and models')
    parse.add_argument('--resume_from', help='the dir of the pretrained model')
    parse.add_argument('--evaluate', action='store_true', help='only evaluate')
    return parse.parse_args()


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 读取命令行参数
    args = parse_args()
    print('training config:', args)

    # 读取配置文件
    cfg = Config.fromfile(args.config_file)
    mmcv.mkdir_or_exist(cfg.work_dir)

    # 设置随机种子
    manualSeed = 10 #cfg.manualseed
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 配置模型和loss
    model = Model(n_channels=cfg.nc, bilinear=cfg.bilinear)
    criterion = torch.nn.MSELoss()#squared l2 loss

    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr)

    # 打开日志文件
    log = log_txt(path=cfg.work_dir, description=cfg.exp)

    # 加载预训练参数
    start_epoch = 0
    if cfg.resume_from is not None:
        print('loading pretrained model from %s' % cfg.resume_from)
        checkpoint = torch.load(cfg.resume_from)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        start_epoch = checkpoint['epoch']


    # 训练和验证模式
    dataloader = load_data(cfg)
    train_loader = dataloader['train']
    test_loader = dataloader['test']
    train4val_loader = dataloader['train4val']

    # 配置训练策略
    iter_per_epoch = len(train_loader)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.milestone,
                                                     gamma=0.5)

    for i in range(0, cfg.epoch_size):
        train_epoch(i, model, train_loader, criterion, optimizer, cfg, log, train_scheduler)
        test_epoch(model, train4val_loader, test_loader, L1_measure, cfg, log, i)


