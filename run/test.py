from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import random_split

import argparse
import os
import pprint
import logging
import json
import torchvision

import _init_paths
import dataset

from utils.utils import create_logger, load_model_state
from core.config import config, update_config
from core.function import train, validate
from models.aar_net import AARNet
from dataset.aar_dataset import AARDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.cfg is not None:
        update_config(args.cfg)

    return args


def main():
    args = parse_args()

    # Log Manager
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    print('=> Loading data ..')
    gpus = [int(i) for i in config.GPUS.split(',')]

    # Test Dataset
    test_dataset = AARDataset(config, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info('=> dataloader length : test({})'.format(test_loader.__len__()))

    # Cudnn Setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    torch.autograd.set_detect_anomaly(True)

    # Model Construction
    print('=> Constructing models ..')
    model = AARNet(config)

    # Model Parallelization
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Load Best State
    model = load_model_state(model, final_output_dir, filename='best_model.pth.tar')

    # Main Prediction Loop
    print('=> Predict...')
    validate(config, model, test_loader, final_output_dir)


if __name__ == '__main__':
    main()