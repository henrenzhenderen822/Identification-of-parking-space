from matplotlib import pyplot as plt
import numpy as np

# # 刻度范围（11组）
# l = list(range(11))
# 刻度值列表
reslution = [32, 25, 20, 16, 12, 9, 7, 5, 4, 3, 2]

# 下面数据是10.3号使用两层卷积50网络在CNRPark上训练的结果
# val1 = [96.45, 96.60, 96.56, 96.45, 96.46, 96.31, 95.88, 95.54, 95.06, 93.63, 88.89]
# val2 = [87.89, 88.50, 87.65, 87.65, 88.16, 90.00, 89.75, 85.56, 86.63, 79.32, 62.64]
# val3 = [78.00, 79.00, 79.00, 77.00, 78.00, 79.00, 79.00, 76.00, 79.00, 77.00, 70.00]

# 下面数据是10.3号使用两层卷积50网络在PKLot上训练的结果
val1 = [85.22, 84.84, 84.39, 84.52, 83.55, 83.05, 81.95, 82.56, 82.90, 82.77, 73.47]
val2 = [99.27, 99.30, 99.32, 99.35, 99.36, 99.32, 99.28, 99.22, 98.93, 98.30, 93.02]
val3 = [86.00, 85.00, 89.00, 87.00, 88.00, 86.00, 86.00, 84.00, 84.00, 84.00, 62.00]
val = []
for i in range(len(reslution)):
    val.append((val2[i]+val3[i])/2)
print(val)

plt.figure(1)
plt.title('(train on CNRPark)')
plt.xlabel('size')
plt.ylabel('accuracy')
plt.xticks(reslution)      # 设置x轴刻度
plt.gca().invert_xaxis()   # x轴逆序
plt.plot(reslution, val1, label="CNRPark", marker='^')
plt.plot(reslution, val2, label="PKLot", marker='v')
plt.plot(reslution, val3, label="mydata", marker='s')
plt.legend(['CNRPark', 'PKLot', 'mydata'])
plt.show()

plt.figure(2)
plt.title('(train on CNRPark)')
plt.xlabel('size')
plt.ylabel('accuracy')
plt.xticks(reslution)      # 设置x轴刻度
plt.gca().invert_xaxis()   # x轴逆序
plt.xticks(reslution)    # 设置x轴刻度
plt.plot(reslution, val, label='average', marker='o')
plt.show()

