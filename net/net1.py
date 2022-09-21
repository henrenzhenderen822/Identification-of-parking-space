'''此程序构建网络适用于1×1大小'''

import torch
from torch import nn


# 因为测试时是用一张一张图片测的，所以加入BatchNorm层的神经网络在测试时要写model.eval()从而把batchnorm固定住，否则效果很差
class Binarynet(nn.Module):

    def __init__(self):
        super(Binarynet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(3, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 2)
        )

    def forward(self, x):
        # print(x.shape)
        x = x.contiguous().view(x.shape[0], -1)   # 打平操作
        x = self.fc(x)
        return x


# 测试
def main():
    x = torch.randn(2, 3, 1, 1)
    model = Binarynet()
    print(model)
    pred = model(x)
    print(pred.shape)

    print("模型的参数量为: {}  ".format(sum(x.numel() for x in model.parameters())))


if __name__ == '__main__':
    main()