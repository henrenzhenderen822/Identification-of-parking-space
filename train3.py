'''此程序与train2.py相同，用plot可视化'''

import torch
from torch import nn, optim
import os
import time
from datasets.datasets1 import train_loader, test_loader   # 直接加载构建好的数据集
from datasets.datasets3 import test_loader as test_loader2
from datasets.datasets3 import train_loader as train_loader2
from parameters import Parameters
from matplotlib import pyplot as plt
import numpy as np


# 设置随机数种子，确保每次的初始化相同
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 主要参数
epochs = Parameters.epochs
leaning_rate = Parameters.learning_rate
img_size = Parameters.img_size

'''★★★★★此处修改模型的保存路径★★★★★'''
path = 'checkpoints/c2p/'

device = torch.device('cuda')
model = Parameters.model.to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=leaning_rate)     # 优化器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)   # 学习率每4个epoch衰减成原来的1/2。

start_time = time.time()


# 训练、验证
def main():

    print(img_size)
    _loss = []
    _global_step = []
    _epochs = []
    _val1 = []
    _val2 = []
    # 记录开始训练时间
    since = time.time()

    now = time.strftime('%m-%d_%H%M')  # 结构化输出当前的时间
    best_acc = 0
    global_step = 0
    last_epoch = 0
    for epoch in range(epochs):
        # 训练
        model.train()
        for batchidx, (x, label) in enumerate(train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)  # 损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 绘制loss曲线（每10个epoch）
            if global_step % 10 == 0:
                _loss.append(loss.item())
                _global_step.append(global_step)
            global_step += 1

        # print('当前学习率为：{}'.format(scheduler.get_last_lr()))
        # scheduler.step()  # 学习率衰减标识
        # 打印loss值，由于loss下降很快，所以利用loss曲线观察更加合适。
        # print('Epoch: {}  loss: {:.4f}'.format(epoch, loss.item()))  # 用.item把tensor转化为numpy

        # 验证
        model.eval()
        with torch.no_grad():  # 表示测试过程不需要计算梯度信息

            '''在CNRPark测试集上验证'''
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
                total_num += x.size(0)
            epoch_acc = total_correct / total_num  # 准确度
            _val1.append(epoch_acc)
            print('Epoch: {}  CNRPark_test_acc: {:.2f}%'.format(epoch, epoch_acc*100))

            '''在PKLot部分数据集上验证'''
            total_correct2 = 0
            total_num2 = 0
            for x, label in test_loader2:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct2 += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
                total_num2 += x.size(0)
            epoch_acc2 = total_correct2 / total_num2  # 准确度
            _val2.append(epoch_acc2)
            print('Epoch: {}  PKLot_test_acc: {:.2f}%'.format(epoch, epoch_acc2 * 100))

            _epochs.append(epoch)

            # 保存验证效果好的模型
            if epoch_acc2 > best_acc:
                best_acc = epoch_acc2
                state = {
                    'state_dict': model.state_dict(),    # 模型参数★★★★★★
                    'best_acc': best_acc,        # 最大准确率
                    'optimizer': optimizer.state_dict(),       # 模型优化器
                    'model_struct': model,    # 模型结构
                    'learning_rate': leaning_rate
                }
                # torch.save(state, os.path.join(path, now + '.pth'))    # 以时间命名模型保存下来
                last_epoch = epoch

            # 这里是数据集最后一轮，如果不足10张则展示的数量是实际最后一轮图片数量
            # viz.images(x[:10], nrow=5, win='valid', opts=dict(title='vaild'))
            # viz.text(str(pred[:10].detach().cpu().numpy()), win='pred_label', opts=dict(title='pred_label'))

            if epoch >= 10 and epoch - last_epoch > 3:
                break           # 如果已经经历了10个epoch 且 准确度在最后面4个epoch内没有提升则结束循环

    time_elapsed = time.time() - since
    print('训练总用时: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # plt画出loss曲线
    plt.figure(1)
    plt.title('loss_' + str(img_size))
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.plot(_global_step, _loss)
    plt.savefig('result_line/' + now + '_loss_' + str(img_size) + '.png')
    plt.close()
    # plt画出准确率曲线图
    plt.figure(2)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(_epochs, _val1, label="CNRPark", marker='^')
    plt.plot(_epochs, _val2, label="PKLot", marker='v')
    plt.savefig('result_line/' + now + '_val_' + str(img_size) + '.png')
    plt.close()

if __name__ == '__main__':
    for i in [5, 10, 20, 30, 40, 50]:
        print()
        epochs = epochs
        leaning_rate = leaning_rate
        img_size = i
        '''★★★★★模型的保存路径★★★★★'''
        path = path + str(img_size)
        main()

    total_time = time.time() - start_time
    print('总用时: {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))