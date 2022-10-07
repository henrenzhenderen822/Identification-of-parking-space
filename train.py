'''此程序用来训练网络(使用visdom可视化)'''

import torch
from torch import nn, optim
import os
import visdom
import time
from datasets.datasets1 import train_loader, test_loader   # 直接加载构建好的数据集
from datasets.datasets2 import train_loader as train_loader2
from datasets.datasets2 import test_loader as test_loader2
from datasets.datasets5 import my_loader
from parameters import Parameters
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

'''★★★★★此处修改模型的保存路径★★★★★'''
path = 'checkpoints/c2p/'

device = torch.device('cuda')
model = Binarynet().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=leaning_rate)     # 优化器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)   # 学习率每4个epoch衰减成原来的1/2。

start_time = time.time()


# 训练、验证
def main():

    # 记录开始训练时间
    since = time.time()

    viz = visdom.Visdom()
    viz.line([0], [0], win='loss'+str(img_size), opts=dict(title='loss'+str(img_size)))
    viz.line([[0.5], [0.5], [0.5]], [0], win='val_acc'+str(img_size), opts=dict(title='val_acc_'+str(img_size), legend=['CNRPark', 'PKLot', 'mydata']))
    now = time.strftime('%m-%d_%H%M' + '.pth')  # 结构化输出当前的时间
    best_acc = 0
    global_step = 0
    last_epoch = 0
    for epoch in range(epochs):

        for batchidx, (x, label) in enumerate(train_loader):
            # 训练
            model.train()
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)  # 损失函数

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 20 == 0:
                # 绘制loss曲线（每10个epoch）
                viz.line([loss.item()], [global_step], win='loss' + str(img_size), opts=dict(title='loss' + str(img_size)), update='append')
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
                    print('total_batch: {}  CNRPark_test_acc: {:.2f}%'.format(global_step, acc*100))

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
                    print('total_batch: {}  PKLot_test_acc: {:.2f}%'.format(global_step, acc2 * 100))

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
                    print('total_batch: {}  mydata_test_acc: {:.2f}%'.format(global_step, acc3 * 100))

                    # 保存在PKLot数据集上准确度最高的模型
                    # if epoch_acc2 > best_acc:
                    #     best_acc = epoch_acc2
                    #     state = {
                    #         'state_dict': model.state_dict(),    # 模型参数★★★★★★
                    #         'best_acc': best_acc,        # 最大准确率
                    #         'optimizer': optimizer.state_dict(),       # 模型优化器
                    #         'model_struct': model,    # 模型结构
                    #         'learning_rate': leaning_rate
                    #     }
                    #     # torch.save(state, os.path.join(path, now))    # 以时间命名模型保存下来
                    #     last_epoch = epoch

                    viz.line([[acc], [acc2], [acc3]], [global_step], win='val_acc'+str(img_size), update='append')
                    # 这里是数据集最后一轮，如果不足10张则展示的数量是实际最后一轮图片数量
                    # viz.images(x[:10], nrow=5, win='valid', opts=dict(title='vaild'))
                    # viz.text(str(pred[:10].detach().cpu().numpy()), win='pred_label', opts=dict(title='pred_label'))
            global_step += 1

    time_elapsed = time.time() - since
    print('总用时: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    print('当前图像大小为：{}'.format(img_size))
    print('当前学习率为：{}'.format(leaning_rate))
    '''★★★★★模型的保存路径★★★★★'''
    path = path + str(img_size)
    main()


