'''此程序使用的是自己整理的 mydata 这个小数据集'''

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import random
from parameters import Parameters


'''图像大小以及批大小'''
size = Parameters.img_size
batchsz = 10
input_size = Parameters.input_size


# 数据处理
data_transforms = transforms.Compose([transforms.Resize([size, size]),
                                      transforms.Resize([input_size, input_size]),
                                      transforms.ToTensor()])

my_dataset = datasets.ImageFolder('data/mydata', data_transforms)
my_loader = DataLoader(my_dataset, batch_size=batchsz, shuffle=True)


def main():
    print('数据集数量为:', len(my_loader.dataset))

    x, y = next(iter(my_loader))
    print(x.shape, y.shape)


if __name__ == '__main__':
    main()
