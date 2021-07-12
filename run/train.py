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

from utils.utils import save_checkpoint, load_checkpoint, create_logger
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


def get_optimizer(model):
    lr = config.TRAIN.LR
    for params in model.module.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    return model, optimizer


def main():
    args = parse_args()

    # Log Manager
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    print('=> Loading data ..')
    gpus = [int(i) for i in config.GPUS.split(',')]

    # Training and Validation Dataset
    total_dataset = AARDataset(config, is_train=True)
    num_data = total_dataset.__len__()
    num_valid = int(num_data * config.TRAIN.VALIDATION_RATIO)
    num_train = num_data - num_valid

    train_dataset, valid_dataset = random_split(total_dataset, [num_train, num_valid])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_patch_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    logger.info('=> dataloader length : train({}), valid({})'.format(
        train_loader.__len__(), valid_loader.__len__()))

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

    # Optimizer Setting
    model, optimizer = get_optimizer(model)

    # Epoch Setting
    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_precision = 0

    # Load Past State
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Main Training Loop
    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        train(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        precision = validate(config, model, test_loader, final_output_dir)

        # Renew Best State
        if precision > best_precision:
            best_precision = precision
            best_model = True
        else:
            best_model = False

        # Save Checkpoint
        logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'precision': best_precision,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    # Save Final Checkpoint
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()