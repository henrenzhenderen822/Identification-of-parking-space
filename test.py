import os
import random
import torch
from torchvision import transforms
import numpy as np
import cv2
import visdom
import time
from matplotlib import pyplot as plt
from datasets.datasets5 import my_loader
from net2 import net20
from parameters import Parameters


viz = visdom.Visdom()
x, label = next(iter(my_loader))
viz.images(x, nrow=5, win='test', opts=dict(title='test'))
viz.text(str(label.numpy()), win='h', opts=dict(title='h'))


# 加载模型
modelname = 'checkpoints/c2p/20/09-22_1134.pth'  # 已保存的模型文件
model = Parameters.model.to('cuda')
model.load_state_dict(torch.load(modelname)['state_dict'])   # 加载保存好的模型
device = 'cuda'

model.eval()
with torch.no_grad():  # 表示测试过程不需要计算梯度信息

    '''在mydata数据集上验证'''
    total_correct = 0
    total_num = 0
    for x, label in my_loader:
        x, label = x.to(device), label.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total_correct += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
        total_num += x.size(0)
    print(total_correct)
    print(total_num)
    accuracy = total_correct / total_num  # 准确度

print(accuracy)















