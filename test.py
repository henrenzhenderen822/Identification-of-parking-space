import os
import random
import torch
from torchvision import transforms
import numpy as np
import cv2
import visdom
import time


m = torch.load('checkpoints_c2p_2/40/09-11_1501.pth')['model_struct']
print(m)
acc = torch.load('checkpoints_c2p_2/40/09-11_1501.pth')['best_acc']
print(acc)

