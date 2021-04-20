# @Author : 游永全
# @Date : 2021/4/20
# @Edition : Python3
# @System : Windows

print('@Author : 游永全')
print('@Date : 2021/4/20')
print('@Edition : Python3')
print('@System : Windows')
print('-' * 50)


def close():
    print('程序即将关闭！')
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)
    exit()


def main(hh=100, hl=90, th=40, tl=0):
    root = os.listdir()
    for i in root:
        if os.path.isfile(i):
            head, end = os.path.splitext(i)
        else:
            continue
        if end == '.csv':
            with open(i, 'r', encoding='utf-8') as f:
                df = pd.read_csv(f)
                x = df.index
                y_1 = df['Humidity(%)']
                y_2 = df['Temperature(℃)']
                plt.subplot(2, 1, 1)
                plt.title(head)
                plt.ylabel("Humidity(%)")
                plt.ylim((hl, hh))
                plt.grid()
                plt.plot(x, y_1)
                plt.subplot(2, 1, 2)
                plt.xlabel("Hour(H)")
                plt.ylabel("Temperature(℃)")
                plt.ylim((tl, th))
                plt.grid()
                plt.plot(x, y_2)
                plt.savefig(head + '.jpg', dpi=400)
                print(i, '可视化图片已生成')
        else:
            continue
    close()


try:
    import os
    import time
    import matplotlib.pyplot as plt
    import pandas as pd
except ModuleNotFoundError:
    print('未安装所需模块：matplotlib, pandas')
    print('请安装后再运行本程序')
    close()
print('本程序自动获取同目录下的温湿度记录数据，并生成对应的折线图文件。')
try:
    low = int(input('请输入湿度下限（0~90）：'))
    if not 0 <= low <= 90:
        raise ValueError
    main(hl=low)
except ValueError:
    print('湿度下限无效！')
    close()
