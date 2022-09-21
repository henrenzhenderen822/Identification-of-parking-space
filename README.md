# -实验探究多种低分辨停车场车位图像的识别
# -pytorch版本1.10.2   cuda版本10.2    opencv版本4.5.5
# cameras文件夹建立了停车场图像与对应的停车位坐标
# checkpoints文件夹用于保存训练好的模型
# datasets文件夹用于保存数据集以及对应的标签（txt格式）
# net以及net2文件夹保存了各种分辨率输入下的网络结构
# result文件夹用于保存测试图片的结果
# dataset.py文件用于构建数据集
# mark.py文件用于标记并返回停车位的坐标（图片及返回结果都保存到了cameras文件夹中）
# run.py文件用于测试（其结果保存在result文件夹中）
# train.py用于训练网络
