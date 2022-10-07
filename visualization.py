'''此程序用于卷积神经网络特征图的可视化'''

import torch
import os
import cv2
import numpy as np
import time
from parameters import Parameters
import torchsummary
from datasets.datasets1 import train_loader, test_loader
import torchvision.utils as vutils

# model_folder = 'checkpoints/train_on_CNRPark/4'
IMAGE_FOLDER = 'visualization'
INSTANCE_FOLDER = None

# 加载模型
modelname = 'checkpoints/train_on_CNRPark/32/09-29_0555.pth'  # 已保存的模型文件
model = Parameters.model.to('cuda')
model.load_state_dict(torch.load(modelname)['state_dict'])   # 加载保存好的模型


def hook_func(module, input, output):
    """
    Hook function of register_forward_hook

    Parameters:
    -----------
    module: module of neural network
    input: input of module
    output: output of module
    """
    image_name = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    vutils.save_image(data, image_name, pad_value=0.5)


def get_image_name_for_hook(module):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(INSTANCE_FOLDER, '%s_%d.png' % (base_name, index))
    return image_name


model.eval()
modules_for_plot = (torch.nn.ReLU, torch.nn.Conv2d)
for name, module in model.named_modules():
    if isinstance(module, modules_for_plot):
        module.register_forward_hook(hook_func)

'''batch_size设置为1'''
index = 1
for data, classes in test_loader:
    INSTANCE_FOLDER = os.path.join(IMAGE_FOLDER, '%d-%d' % (index, classes.item()))
    data, classes = data.cuda(), classes.cuda()
    outputs = model(data)

    index += 1
    if index > 20:
        break
