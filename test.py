# import os
#
# import torch
# import torchvision.models as models
# import torch.nn as nn
# import timm
# from config import cfg
#
#
# model = timm.create_model(cfg['model'], pretrained=False)
# model.classifier = nn.Linear(model.classifier.in_features, 20)
#
# batch_size = 1  # 批处理大小
# input_shape = (3, 244, 384)  # 输入数据,改成自己的输入shape
#
# # #set the model to inference mode
# model.eval()
#
# x = torch.randn(batch_size, *input_shape)  # 生成张量
# export_onnx_file = "test.onnx"  # 目的ONNX文件名
# torch.onnx.export(model,
#                   x,
#                   export_onnx_file,
#                   opset_version=11,
#                   do_constant_folding=True,  # 是否执行常量折叠优化
#                   input_names=["input"],  # 输入名
#                   output_names=["output"],  # 输出名
#                   dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
#                                 "output": {0: "batch_size"}})

import numpy as np
bounds = [[0.47859862, 0.46757355, 0.39008468], [0.2554969, 0.2480743, 0.25745383]]
a = [1, 1, 1]

bounds=np.array(bounds)
a=np.array(a)

print((a-bounds[0])/bounds[1])