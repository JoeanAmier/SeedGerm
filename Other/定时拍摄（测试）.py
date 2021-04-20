# @Author : 游永全
# @Date : 2021/4/20
# @Edition : Python3
# @System : Raspberry_Pi
import os
import smbus
import time
from picamera import PiCamera
import csv


def start(_name, _experiment, number):
    with PiCamera() as camera:
        camera.resolution = (2592, 1944)
        camera.framerate = 15
        root = os.getcwd() + '/' + _experiment + '/'
        if not os.path.exists(root):
            os.mkdir(root)
        number += 1
        progress = 0
        while True:
            camera.start_preview()
            time.sleep(5)
            date = time.strftime("%d-%m-%Y_%H-%M", time.localtime(time.time()))
            path = root + _experiment + '_ID-' + \
                   str(progress) + '_Data-' + date + '.jpg'
            camera.capture(path)
            camera.stop_preview()
            print(
                '已拍摄图像：' +
                _experiment +
                '_ID-' +
                str(progress) +
                '_Data-' +
                date)
            sensor(_name)
            progress += 1
            if progress == number:
                break
            time.sleep(5)
        print('定时拍摄结束')


def sensor(_name):
    bus = smbus.SMBus(1)
    bus.write_i2c_block_data(0x44, 0x2C, [0x06])
    time.sleep(5)
    data = bus.read_i2c_block_data(0x44, 0x00, 6)
    temp = data[0] * 256 + data[1]
    c_temp = -45 + (175 * temp / 65535.0)
    humidity = 100 * (data[3] * 256 + data[4]) / 65535.0
    t = time.strftime("%d-%m-%Y_%H-%M", time.localtime(time.time()))
    h = '%.2f' % humidity
    c = '%.2f' % c_temp
    with open(_name, 'a', encoding='utf-8', newline='')as _f:
        _row = [t, h, c]
        _write = csv.writer(_f)
        _write.writerow(_row)
    print('时间：', t, end='，')
    print("湿度：", h, "%", end='，')
    print("温度：", c, "℃")


def main():
    try:
        print('根据提示输入实验信息，输入后回车\n若信息输入错误请按 Ctrl + C 或 Stop，然后重新运行本程序')
        data1 = input('输入材料名称：（如输入：OSR96）')
        data2 = input('输入处理浓度：（如输入：0.6Mpa）')
        data3 = 'Rep' + input('输入重复数：（如输入：6）')
        data4 = 'CAM' + input('输入摄像头编号：（如输入：26）')
        data5 = int(input('输入拍摄图像数量：（如输入：12）'))
        enter = input('检查输入信息，无误请输入Y，有误请输入N\n')
        if enter not in ['Y', 'N'] or enter != 'Y':
            raise KeyboardInterrupt
        experiment = data1 + '-' + data2 + '_' + data3 + '_' + data4
    except KeyboardInterrupt:
        print('请重新运行本程序')
        exit()
    name = '%s.csv' % experiment
    with open(name, 'w', encoding='utf-8', newline='') as f:
        row = ['Time', 'Humidity(%)', 'Temperature(℃)']
        write = csv.writer(f)
        write.writerow(row)
    print('实验开始，请不要关闭此窗口\n本程序自动记录温湿度数据到CSV文件，实验时请不要打开CSV文件以免记录失败')
    try:
        start(name, experiment, data5)
    except KeyboardInterrupt:
        print('实验已终止')
        exit()
    print('实验结束')


if __name__ == '__main__':
    main()
