'''此程序建立了CNRPark数据集 中的所有patch以及对应label的数据集 （CNRPark-Ext数据集）'''

import torch
import os
from torch.utils.data import Dataset, DataLoader  # Dataset:自定义数据集的母类
from torchvision import transforms  # 图片的变换器
from PIL import Image  # PIL(Python Image Library)是python的第三方图像处理库
import pandas as pd
import random
from parameters import Parameters


'''图像大小以及批大小'''
size = Parameters.img_size
batchsz = Parameters.batch_size


class Patchs(Dataset):

    def __init__(self, root, tf, mode):   # mode:模式（训练 or 测试）   tf:transform对图像采取变换
        super(Patchs, self).__init__()

        self.root = root
        self.tf = tf

        # image_path, label   (路径+标签)
        self.images, self.labels = self.load_txt('data/LABELS/all.txt')

        # 从上面的全部图片信息中截取不同的比例用作不同用途
        if mode == 'train':
            self.images = self.images[:30000]
            self.labels = self.labels[:30000]
        elif mode == 'test':
            self.images = self.images[70000:100000]
            self.labels = self.labels[70000:100000]

    def load_txt(self, filename):

        '''读取（加载）txt文件'''
        images, labels = [], []
        with open(filename, 'r', encoding='utf-8') as file:
            l = file.readlines()  # readlines 是一个列表，它会按行读取文件的所有内容
            # random.shuffle(l)     # 将txt中的信息按行打乱，但仍然对应正确的label

        for i in range(len(l)):
            image, label = l[i].split(' ')
            images.append(self.root + '/' + image)
            labels.append(int(label))

        assert len(images) == len(labels)  # 确保图片和标签的列表长度一致，不一致会报错
        return images, labels

    def __len__(self):
        return len(self.images)    # 裁剪过后的长度

    def __getitem__(self, idx):
        # idx： [0 - len(images)]
        # self.images, self.labels
        # 图片信息目前不是想要的数据类型（需要转化为图片信息）
        # label: 0,1 标签信息已经是数据类型了
        img, label = self.images[idx], self.labels[idx]
        # print(img, label)

        img = self.tf(img)    # 变为数据
        label = torch.tensor(label)   # 把label也变为tensor类型

        return img, label


# 对训练集数据进行0.5概率的水平翻转（左右镜像）
data_transforms ={
    'train': transforms.Compose([
        lambda x:Image.open(x).convert('RGB'),  # string path => image data (变为图像的数据类型)
        transforms.RandomHorizontalFlip(0.5),  # 水平角度翻转
        # transforms.RandomRotation(10),  # 随机旋转 +-10度
        # transforms.Resize([60]),  # 重新设置大小
        # transforms.RandomCrop([50, 50]),  # 裁剪成50×50
        transforms.Resize([size, size]),
        transforms.ToTensor()]),
    'test': transforms.Compose([
        lambda x:Image.open(x).convert('RGB'),  # string path => image data (变为图像的数据类型)
        transforms.Resize([size, size]), transforms.ToTensor()])
    }

train_datasets = Patchs('data/PATCHES', data_transforms['train'], mode='train')
test_datasets = Patchs('data/PATCHES', data_transforms['test'], mode='test')

train_loader = DataLoader(train_datasets, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=batchsz, shuffle=True)


def main():
    # 调用时会产生相对路径问题，若要在此处运行，则要修改文件路径为'../data'
    a = len(train_loader.dataset)
    b = len(test_loader.dataset)
    c = a + b
    print('训练集的数量为：', a)
    print('测试集的数量为：', b)
    print('全部数据集数量为：', c)

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)


if __name__ == '__main__':
    main()

