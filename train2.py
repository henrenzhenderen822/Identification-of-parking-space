'''此程序功能与train.py相同，用plot可视化'''

import torch
from torch import nn, optim
import os
import time
from datasets.datasets1 import train_loader, test_loader   # 直接加载构建好的数据集
from datasets.datasets2 import train_loader as train_loader2
from datasets.datasets2 import test_loader as test_loader2
from datasets.datasets5 import my_loader
from parameters import Parameters
from matplotlib import pyplot as plt
import numpy as np
from net.net_2conv import Binarynet    # 选择训练的模型


# 设置随机数种子，确保每次的初始化相同
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 主要参数
epochs = Parameters.epochs
leaning_rate = Parameters.learning_rate
img_size = Parameters.img_size

'''-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-★★★★★训练集的选择★★★★★-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-'''
option = 'train_on_PKLot'
'''模型的保存路径'''
path = 'checkpoints/' + option + '/' + str(img_size)
'''★★★★★训练网络的描述★★★★★'''
model_kind = '两层卷积'

# 其他配置
device = torch.device('cuda')
model = Binarynet().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=leaning_rate)  # 优化器(AdamW 即 Adam + weight decay, 效果与 Adam + L2正则化相同)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)   # 学习率每4个epoch衰减成原来的1/2。


# 训练、验证
def main():

    loss_list = []
    global_list = []
    val1_list = []
    val2_list = []
    val3_list = []
    # 记录开始训练时间
    since = time.time()

    now = time.strftime('%m-%d_%H%M')  # 结构化输出当前的时间
    best_acc = 0
    global_step = 0
    last_epoch = 0
    for epoch in range(epochs):

        for batchidx, (x, label) in enumerate(train_loader2):
            # 训练
            model.train()
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)  # 损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 20 == 0:
                # loss值（每10个epoch）
                loss_list.append(loss.item())
                global_list.append(global_step)
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
                    acc = total_correct / total_num  # 准确度
                    val1_list.append(acc)
                    print('global_step: {}  CNRPark_test_acc: {:.2f}%'.format(global_step, acc*100))
                    # with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
                    #     f.write('Epoch: {}  CNRPark_test_acc: {:.2f}%\n'.format(epoch, acc*100))

                    '''在PKLot部分数据集上验证'''
                    total_correct2 = 0
                    total_num2 = 0
                    for x, label in test_loader2:
                        x, label = x.to(device), label.to(device)
                        logits = model(x)
                        pred = logits.argmax(dim=1)
                        total_correct2 += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
                        total_num2 += x.size(0)
                    acc2 = total_correct2 / total_num2  # 准确度
                    val2_list.append(acc2)
                    print('global_step: {}  PKLot_test_acc: {:.2f}%'.format(global_step, acc2 * 100))
                    # with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
                    #     f.write('Epoch: {}  PKLot_test_acc: {:.2f}%\n'.format(epoch, acc2 * 100))

                    '''在 mydata数据集 上验证'''
                    total_correct3 = 0
                    total_num3 = 0
                    for x, label in my_loader:
                        x, label = x.to(device), label.to(device)
                        logits = model(x)
                        pred = logits.argmax(dim=1)
                        total_correct3 += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
                        total_num3 += x.size(0)
                    acc3 = total_correct3 / total_num3  # 准确度
                    val3_list.append(acc3)
                    print('global_step: {}  mydata_test_acc: {:.2f}%'.format(global_step, acc3 * 100))
                    # with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
                    #     f.write('Epoch: {}  mydata_test_acc: {:.2f}%\n'.format(epoch, acc3 * 100))

                    # 保存某一轮训练后在另一数据集效果最好的模型
                    # if epoch_acc2 > best_acc:
                    #     best_acc = epoch_acc2
                    #     state = {
                    #         'state_dict': model.state_dict(),    # 模型参数★★★★★★
                    #         'best_acc': best_acc,        # 最大准确率
                    #         'optimizer': optimizer.state_dict(),       # 模型优化器
                    #         'model_struct': model,    # 模型结构
                    #         'learning_rate': leaning_rate
                    #     }
                    #     torch.save(state, os.path.join(path, now + '.pth'))    # 以时间命名模型保存下来
                    #     last_epoch = epoch

                    # if epoch >= 10 and epoch - last_epoch > 3:
                    #     break           # 如果已经经历了10个epoch 且 准确度在最后面4个epoch内没有提升则结束循环
            global_step += 1

    # 保存最后一轮训练完成后的网络模型
    state = {
        'state_dict': model.state_dict(),       # 模型参数
        'optimizer': optimizer.state_dict(),    # 模型优化器
        'model_struct': model,                  # 模型结构
    }
    torch.save(state, os.path.join(path, now + '.pth'))  # 以时间命名模型保存下来

    time_elapsed = time.time() - since
    print('本次实验用时: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
    #     f.write('本次实验用时: {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    # plt画出loss曲线
    plt.figure(1)
    plt.title('loss ' + str(img_size))
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.plot(global_list, loss_list)
    plt.savefig('result_line/' + now + '_loss_' + str(img_size) + '_' + str(leaning_rate) + '.png')
    plt.close()
    # plt画出准确率曲线图
    plt.figure(2)
    plt.title('acc' + '(' + option + ')' + str(img_size))
    plt.xlabel('batch')
    plt.ylabel('acc')
    # l = np.linspace(0, 70, 15)
    # plt.xticks(l)  # 设置x轴刻度
    plt.plot(global_list, val1_list, label="CNRPark")   # marker='^'
    plt.plot(global_list, val2_list, label="PKLot")     # marker='v'
    plt.plot(global_list, val3_list, label="mydata")    # marker='s'
    plt.legend(['CNRPark', 'PKLot', 'mydata'])
    plt.savefig('result_line/' + now + '_val_' + str(img_size) + '_' + str(leaning_rate) + '.png')
    plt.close()


if __name__ == '__main__':

    print('当前图像大小为：{}'.format(img_size))
    with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
        f.write('当前图像大小为：{}\n'.format(img_size))
    print('当前学习率为：{}'.format(leaning_rate))
    with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
        f.write('当前学习率为：{}\n'.format(leaning_rate))
    print('训练的模型结构为：' + model_kind)
    with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
        f.write('训练的模型结构为：' + model_kind + '\n')
    main()


