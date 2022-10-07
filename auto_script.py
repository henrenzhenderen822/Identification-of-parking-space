'''此程序为自动化运行脚本，方便自动修改参数以训练网络'''

import datetime
import os
import threading
import time


def execCmd(cmd):
    try:
        print("命令%s开始运行%s" % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("命令%s结束运行%s" % (cmd, datetime.datetime.now()))
    except:
        print('%s\t 运行失败' % (cmd))


if __name__ == '__main__':

    start_time = time.time()

    # 是否需要并行运行
    if_parallel = False   # 显存较小，串行比较合适

    # 需要执行的命令列表
    img_list = ['2', '3', '4', '5', '7', '9', '12', '16', '20', '25', '32']
    cmds = ['python train2.py --batch_size=32 --epochs=1000 --learning_rate=2e-5 --input_size=50 --img_size=' + i for i in img_list]

    with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n★★★自动化脚本运行★★★\n本次实验使用统一的50×50输入网络结构\n')
    if if_parallel:
        # 并行
        threads = []
        for cmd in cmds:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)

        # 等待线程运行完毕
        for th in threads:
            th.join()
    else:
        # 串行
        for cmd in cmds:
            try:
                print("命令%s 开始运行: %s" % (cmd, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
                with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
                    f.write("命令%s 开始运行: %s\n" % (cmd, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
                os.system(cmd)
                print("命令%s 结束运行: %s\n" % (cmd, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
                with open('result_line/acc.txt', 'a', encoding='utf-8') as f:
                    f.write("命令%s 结束运行: %s\n\n" % (cmd, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
            except:
                print('%s\t 运行失败' % (cmd))

    end_time = time.time()
    total_time = end_time - start_time
    print('总用时: {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    os.startfile('D:/bing_dundun.exe')   # 提醒我程序运行结束了（此行可删）
