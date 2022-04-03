
from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from utils import load_model, AverageMeter, accuracy
import cv2
import json

from config import cfg
from dataset import *
from utils import *


os.chdir("./")   #修改当前工作目录

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

model_file = './tf_efficientnet_b3_ns96.15189873417721.pth.tar'


@torch.no_grad()
def infer(testloader, model):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()

    predicts = []
    img_names = []

    bar = tqdm(enumerate(testloader), total=len(testloader))
    for steps, (inputs, names) in bar:
        inputs = inputs.to('cuda', dtype=torch.float)

        outputs = model(inputs)
        # loss = cross_entropy(outputs, labels)
        # acc = accuracy(outputs, labels)

        # losses.update(loss.item(), inputs.size(0))
        # accs.update(acc[0].item(), inputs.size(0))
        outputs = torch.softmax(outputs, dim=1)
        targets = torch.argmax(outputs,dim=1)

        # # 阈值筛选
        # # 存置信度大于阈值的文件名
        # temp_img_names = []
        # # 存与文件名对应的预测结果
        # temp_predicts = []
        # np_targets = targets.detach().cpu().numpy().tolist()
        # for i in range(cfg['batch_size']):
        #     if outputs[i][np_targets[i]] > 0.98:
        #         temp_img_names.append(names[i])
        #         temp_predicts.append(np_targets[i])
        #
        # # resave predicted images
        # if os.path.exists("./data/validation_images/"):
        #     for i in range(len(temp_img_names)):
        #         if os.path.exists("./data/validation_images/" + str(temp_predicts[i]) + "/"):
        #             if len(os.listdir("./data/validation_images/" + str(temp_predicts[i]) + "/")) < 200:
        #                 source = "./data/test_images/" + str(temp_img_names[i])
        #                 target = "./data/validation_images/" + str(temp_predicts[i]) + '/' + str(temp_img_names[i])
        #                 shutil.copyfile(source, target)
        #         else:
        #             os.mkdir("./data/validation_images/" + str(temp_predicts[i]) + "/")
        #             source = "./data/test_images/" + str(temp_img_names[i])
        #             target = "./data/validation_images/" + str(temp_predicts[i]) + '/' + str(temp_img_names[i])
        #             shutil.copyfile(source,target)

        img_names.extend(names)
        predicts.extend(targets.detach().cpu().numpy().tolist())

        # print(len(img_names))
        # print(len(predicts))

    # submit json
    submit_json = []
    if len(img_names) == len(predicts):
        for i in range(len(img_names)):
            json_item = {}
            json_item["image_id"] = int(img_names[i][:-4])
            json_item["category_id"] = predicts[i]
            submit_json.append(json_item)
    print(submit_json)
    if os.path.exists("./t1_p1_result.json"):
        # 清空文件
        os.remove("./t1_p1_result.json")
        with open("./t1_p1_result.json", "w") as f:
            json.dump(submit_json, f)
        f.close()
    else:
        with open("./t1_p1_result.json", "w") as f:
            json.dump(submit_json, f)
        f.close()
    return losses.avg, accs.avg


testset = MyDataset(data_dir='./data/', mode='test', transform=transform_test)
testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)

# Model
model_dict = torch.load(model_file)
model = load_model(cfg['model'],pretrained=False)
model.load_state_dict(model_dict['state_dict'])
model = model.cuda()


train_loss, train_acc = infer(testloader, model)


