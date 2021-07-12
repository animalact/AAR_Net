from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy
import math
import torch
import numpy as np

from core.evaluate import evaluate

logger = logging.getLogger(__name__)


def train(config, model, optimizer, loader, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Ready for training
    model.train()

    end = time.time()
    for i, (input, target, meta) in enumerate(loader):

        data_time.update(time.time() - end)
        with torch.autograd.set_detect_anomaly(True):

            # Prediction
            pred, loss = model(input, target)

            # Loss backward
            losses.update(loss.item())
            optimizer.zero_grad()
            if loss > 0:
                loss.backward()
            optimizer.step()

            # Time Check
            batch_time.update(time.time() - end)
            end = time.time()

        # Print Log
        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss: {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Memory {memory:.1f}'.format(
                epoch, i, len(loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, model, loader, output_dir):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Ready for evaluation
    model.eval()

    precision_list = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, meta) in enumerate(loader):
            data_time.update(time.time() - end)

            # Prediction
            pred = model(input)

            # Evaluation
            pred = pred.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            precision = evaluate(pred, target)
            precision_list.append(precision)

            # Print Log
            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = 'Test: [{0}/{1}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Memory {memory:.1f}'.format(
                    i, len(loader), batch_time=batch_time,
                    speed=input[0].size(0) / batch_time.val,
                    data_time=data_time, memory=gpu_memory_usage)
                logger.info(msg)

    # Total Evaluation
    mean_precision = sum(precision_list, 0.0) / len(precision_list)
    max_precision = max(precision_list)
    min_precision = min(precision_list)

    msg = '(Patch)\tPrecision: {mean:.4f}\t' \
          'MAX: {max:.4f}\t' \
          'MIN: {min:.4f}'.format(mean=mean_precision, max=max_precision, min=min_precision)
    logger.info(msg)

    return mean_precision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
