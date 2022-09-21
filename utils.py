'''此程序主要用来查看数据集中的图片'''

import torch
import cv2
import numpy as np
from datasets3 import train_loader, test_loader   # 此处更改需要查看的数据集
import matplotlib.pyplot as plt


# 此函数用来展示图片
def cv_show(img, name='test'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 此函数用于把torch图像格式转化为opencv格式
def make_suit_cv(img):
    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR)  # 转成opencv认识的玩意
    img = np.array(img, dtype=np.uint8)

    return img


# 将opencv的GBR格式转化为适合plt的GBR格式即可
def make_suit_plt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# 这里将展示25张（5行5列）数据集中的图像，以供参考
def main():
    x, y = next(iter(test_loader))
    y = y.numpy()

    # 方法一: 用opencv展示图像（这样展示的图片是原始大小，不能手动放大或者缩小）
    image = []
    label = []
    for i in range(25):
        img = make_suit_cv(x[i])
        image.append(img)
        label.append(y[i])

    result1, result2, result3, result4, result5 = image[0], image[5], image[10], image[15], image[20]
    for i in range(1, 5):
        result1 = np.hstack((result1, image[i]))
        result2 = np.hstack((result2, image[5+i]))
        result3 = np.hstack((result3, image[10+i]))
        result4 = np.hstack((result4, image[15+i]))
        result5 = np.hstack((result5, image[20+i]))

    result = np.vstack((result1, result2, result3, result4, result5))

    cv_show(result)
    cv2.imwrite('result.png', result)

    # 方法二：用visdom模块展示图像
    import visdom
    viz = visdom.Visdom()
    viz.images(x[:25], nrow=5, win='show', opts=dict(title='show'))

    # 方法三：用matplotlib中的plot函数展示图像
    # plot的图像是RGB的，而opencv是GBR的，所以要转换通道
    for i in range(25):
        if label[i] == 0:
            z = 'vacant'
        else:
            z = 'busy'
        img = make_suit_plt(image[i])
        plt.subplot(5, 5, i+1)
        plt.title(z)
        if i != 20:
            plt.xticks([]), plt.yticks([])
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
