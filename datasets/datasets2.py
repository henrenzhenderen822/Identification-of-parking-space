'''此程序使用的是CNRpark这个小数据集'''

from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import random
from parameters import Parameters


'''图像大小以及批大小'''
size = Parameters.img_size
batchsz = Parameters.batch_size


# 利用torchvision.datasets中自带的ImageFolder直接构造出数据集
# 划分训练集、测试集 (8：2)
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, mode):
        super(CustomImageFolder, self).__init__(root, transform)
        assert mode in ['train', 'test']
        random.seed(0)
        random.shuffle(self.samples)
        if mode == 'train':
            self.samples = self.samples[:int(0.8*len(self))]
            self.targets = [s[1] for s in self.samples]
            self.imgs = self.samples
        elif mode == 'test':
            self.samples = self.samples[int(0.8*len(self)):]
            self.targets = [s[1] for s in self.samples]
            self.imgs = self.samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


# 数据处理和数据增强
data_transforms ={
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),  # 水平角度翻转
        # transforms.RandomRotation(10),  # 随机旋转 +-15度
        # transforms.Resize([60]),  # 重新设置大小
        # transforms.RandomCrop([50, 50]),  # 随机裁剪
        transforms.Resize([size, size]),
        transforms.ToTensor()]),
    'test': transforms.Compose([transforms.Resize([1, 1]), transforms.ToTensor()])
    }


train_dataset = CustomImageFolder('../data/CNRPark', data_transforms['train'], 'train')
test_dataset = CustomImageFolder('../data/CNRPark', data_transforms['test'], 'test')

train_loader = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsz, shuffle=True)


def main():
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
