'''此程序用来配置参数和网络模型'''
# 实验需要经常更改参数，利用此程序布局配置
import argparse
import sys


parser = argparse.ArgumentParser(description="demo of argparse")
# 通过对象的add_argument函数来增加参数。
parser.add_argument('-b', '--batch_size', default=128, type=int)
parser.add_argument('-i', '--img_size', default='10', type=int)
parser.add_argument('-e', '--epochs', default='18', type=int)
parser.add_argument('-l', '--learning_rate', default='1e-4', type=float)
args = parser.parse_args()
size = args.img_size

# 根据输入图片的大小选择网络结构
if size == 2:
    from net2.net2 import Binarynet
elif size == 3:
    from net2.net3 import Binarynet
elif size == 4:
    from net2.net4 import Binarynet
elif size == 5:
    from net2.net5 import Binarynet
elif size == 6:
    from net2.net6 import Binarynet
elif size == 7:
    from net2.net7 import Binarynet
elif size == 8:
    from net2.net8 import Binarynet
elif size == 9:
    from net2.net9 import Binarynet
elif size == 10:
    from net2.net10 import Binarynet
elif size == 20:
    from net2.net20 import Binarynet
elif size == 30:
    from net2.net30 import Binarynet
elif size == 40:
    from net2.net40 import Binarynet
elif size == 50:
    from net2.net50 import Binarynet


'''Parameters类保存运行所需要的参数等配置(类属性，可以直接通过类名调用)'''
class Parameters():
    batch_size = args.batch_size         # 批大小
    img_size = args.img_size             # 图片大小
    epochs = args.epochs                 # 跑epochs轮数据集
    learning_rate = args.learning_rate   # 学习率大小
    model = Binarynet()                  # 适用于图片大小的模型


def main():
    print(args)
    # vars() 函数返回对象object的属性和属性值的字典对象。
    ap = vars(args)
    print(ap)  # {'name': 'Li', 'age': '21'}
    print(ap['batch_size'])  # Li
    print('Hello {} {}'.format(Parameters.batch_size, Parameters.img_size))  # Hello Li 21

    print("Input argument is %s" %(sys.argv))


if __name__ == '__main__':
    main()
