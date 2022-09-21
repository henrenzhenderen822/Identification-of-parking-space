# 此程序用于打印不同分辨率图片在模型上的运行结果（25张）
from net.net3 import Binarynet
import torch
from datasets3 import test_loader
import matplotlib.pyplot as plt
import cv2
import numpy as np


# 加载模型
modelname = 'checkpoint/3/09-03_1317.pth'  # 已保存的模型文件
model = Binarynet()
model.load_state_dict(torch.load(modelname)['state_dict'])   # 加载保存好的模型


# 此函数用于把torch图像格式转化为opencv格式
def make_suit_cv(img):
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR)  # 转成opencv认识的玩意
    img = np.array(img, dtype=np.uint8)

    return img


# 转化为plt画图：将opencv的GBR格式转化为适合plt的GBR格式即可
def make_suit_plt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():

    x, label = next(iter(test_loader))
    logits = model(x)
    pred = logits.argmax(dim=1)

    label = label.numpy()
    pred = pred.numpy()

    for i in range(25):
        color = 'black'
        img = make_suit_cv(x[i])
        img = make_suit_plt(img)

        if pred[i] == 0:
            z = 'vacant'
        else:
            z = 'busy'

        if pred[i] != label[i]:
            color = 'red'

        plt.subplot(5, 5, i + 1)
        plt.title(z, color=color)
        if i != 20:
            plt.xticks([]), plt.yticks([])
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
