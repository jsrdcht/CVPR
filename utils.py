import torch
import os
import torch.nn.functional as F
# import torchvision
import numpy as np
import timm
import torch.nn as nn


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_model(model_name, pretrained=True):
    model = timm.create_model(model_name, pretrained=pretrained)
    model.classifier = nn.Linear(model.classifier.in_features, 20)
    model.eval()
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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


def cross_entropy(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)
