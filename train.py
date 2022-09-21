'''此程序用来训练网络'''

from net.net5ppp import Binarynet  # 可导入不同的网络结构进行训练
import torch
from torch import nn, optim
import os
import visdom
import time
from datasets1 import train_loader, test_loader   # 直接加载构建好的数据集
from datasets3 import test_loader as test_loader2


# 设置随机数种子，确保每次的初始化相同
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 主要参数
epochs = 18
leaning_rate = 5e-4

'''★★★★★在此处修改模型的保存路径★★★★★'''
path = 'mix_checkpoints/5ppp'

device = torch.device('cuda')
model = Binarynet().to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=leaning_rate)     # 优化器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)   # 学习率每4个epoch衰减成原来的1/2。

# 记录开始训练时间
since = time.time()


# 训练、验证
def main():
    viz = visdom.Visdom()
    viz.line([0], [0], win='loss', opts=dict(title='loss'))
    viz.line([0.95], [0], win='val_acc', opts=dict(title='val_acc'))
    now = time.strftime('%m-%d_%H%M' + '.pth')  # 结构化输出当前的时间
    best_acc = 0
    global_step = 0
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

            if global_step % 10 == 0:
                viz.line([loss.item()], [global_step], win='loss', opts=dict(title='loss'), update='append')
            global_step += 1

        print('当前学习率为：{}'.format(scheduler.get_last_lr()))
        scheduler.step()  # 学习率衰减标识
        print('Epoch: {}  loss: {:.4f}'.format(epoch, loss.item()))  # 用.item把tensor转化为numpy

        # 验证
        model.eval()
        with torch.no_grad():  # 表示测试过程不需要计算梯度信息
            total_correct = 0
            total_num = 0
            for x, label in test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()  # 统计预测对的数量
                total_num += x.size(0)
            epoch_acc = total_correct / total_num  # 准确度
            print('Epoch: {}  test_acc: {:.2f}%'.format(epoch, epoch_acc*100))

            # 保存验证效果好的模型
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }

                torch.save(state, os.path.join(path, now))    # 以时间命名模型保存下来
            viz.line([epoch_acc], [epoch], win='val_acc', opts=dict(title='val_acc'), update='append')
            # 这里是数据集最后一轮，如果不足10张则展示的数量是实际最后一轮图片数量
            viz.images(x[:10], nrow=5, win='valid', opts=dict(title='vaild'))
            viz.text(str(pred[:10].detach().cpu().numpy()), win='pred_label', opts=dict(title='pred_label'))

    time_elapsed = time.time() - since
    print('训练总用时: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    main()
