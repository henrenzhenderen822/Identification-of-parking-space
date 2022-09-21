from net.net50 import Binarynet
import torch
from datasets3 import test_loader   # 利用PKLot数据集中的100000张图片对训练好的模型进行验证


device = torch.device('cuda')
# 加载模型
modelname = 'checkpoints/50/09-01_1637.pth'  # 已保存的模型文件
model = Binarynet().to('cuda')
model.load_state_dict(torch.load(modelname)['state_dict'])   # 加载保存好的模型

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
    acc = total_correct / total_num  # 准确度

print('正确个数为:', int(total_correct))
print('总数为:', int(total_num))
print('准确率为: {:.2f}%'.format(acc * 100))

