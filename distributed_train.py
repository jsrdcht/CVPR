from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import cv2

from config import cfg
from dataset import *
from utils import *
from attack import *

# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
CUDA_VISIBLE_DEVICES = [int(i) for i in CUDA_VISIBLE_DEVICES]
print("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES)
# print("^^^^", torch.cuda.get_device_name(0))
# print("^^^^", torch.cuda.get_device_name(1))

### 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# Use CUDA
use_cuda = torch.cuda.is_available()
# 固定seed，使得每次结果一样
set_seed(11037)


def main():
    args = cfg

    trainset = MyDataset(data_dir='./data', mode='train', transform=transform_train)
    validset = MyDataset(data_dir='./data', mode='validation', transform=transform_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)

    trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=0, sampler=train_sampler)
    validloader = data.DataLoader(validset, batch_size=2 * args['batch_size'], shuffle=True, num_workers=0, sampler=valid_sampler)

    # Model
    model = load_model(model_name=args['model'], pretrained=False)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(local_rank)
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    best_acc = 0  # best test accuracy

    optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
                                                       **args['optimizer_hyperparameters'])
    if args['scheduler_name'] != None:
        scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
                                                                              **args['scheduler_hyperparameters'])
    model = model.cuda()
    # Train and val
    for epoch in range(args['epochs']):
        trainloader.sampler.set_epoch(epoch)
        validloader.sampler.set_epoch(epoch)

        train_loss, train_acc = train(trainloader, model, optimizer, epoch=epoch)
        valid_loss, valid_acc = valid(validloader, model, epoch=epoch)

        print('acc: {}'.format(train_acc))
        print('validation_loss: {}, validation_acc: {}'.format(valid_loss, valid_acc))

        # save model
        if dist.get_rank() == 0:
            best_acc = max(valid_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': valid_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, arch=args['model'] + str(best_acc))

        if args['scheduler_name'] != None:
            scheduler.step()

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    bar = tqdm(enumerate(trainloader), total=len(trainloader))
    for steps, (inputs, labels) in bar:
        labels = labels.to(local_rank)
        inputs = inputs.to(local_rank, dtype=torch.float)

        outputs = model(inputs)
        loss = critirion(outputs, labels)
        acc = accuracy(outputs, labels)

        if cfg['attack']:
            for attack_algo in cfg['attack_algos']:
                attack_loss, attack_accuracy = attack(inputs, model, labels, attack_algo=attack_algo)
                print(attack_accuracy)
                loss += attack_loss * cfg['attack_loss_ratio']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))

        bar.set_postfix(epoch=epoch, train_loss=losses.avg, train_acc=accs.avg,
                        lr=optimizer.state_dict()['param_groups'][0]['lr'])

    return losses.avg, accs.avg


@torch.no_grad()
def valid(validloader, model, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode

    bar = tqdm(enumerate(validloader), total=len(validloader))
    for steps, (inputs, labels) in bar:
        labels = labels.to(local_rank)
        inputs = inputs.to(local_rank, dtype=torch.float)

        outputs = model(inputs)
        loss = critirion(outputs, labels)
        acc = accuracy(outputs, labels)

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))

        bar.set_postfix(epoch=epoch, valid_loss=losses.avg, valid_acc=accs.avg)

    return losses.avg, accs.avg


def save_checkpoint(state, arch):
    filepath = os.path.join(arch + '.pth.tar')
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
