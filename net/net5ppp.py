'''此程序构建网络适用于5×5大小'''

import torch
from torch import nn


# 因为测试时是用一张一张图片测的，所以加入BatchNorm层的神经网络在测试时要写model.eval()从而把batchnorm固定住，否则效果很差
class Binarynet(nn.Module):

    def __init__(self):
        super(Binarynet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(30),
        )

        self.conv = nn.Sequential(self.conv1, self.conv2, self.conv3)

        self.fc = nn.Sequential(
            nn.Linear(30*5*5, 60),
            nn.ReLU(),
            nn.Linear(60, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = x.contiguous().view(x.shape[0], -1)   # 打平操作
        x = self.fc(x)
        return x


# 测试
def main():
    x = torch.randn(2, 3, 5, 5)
    model = Binarynet()
    print(model)
    pred = model(x)
    print(pred.shape)

    print("模型的参数量为: {}  ".format(sum(x.numel() for x in model.parameters())))


if __name__ == '__main__':
    main()
