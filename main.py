#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" germapp.py - Main script for runnning the germination scoring application.

启动核心线程进行繁重的计算处理，启动GUI进行用户交互。

hello world
"""

import matplotlib

matplotlib.use('Agg')
import time
import os

if not os.path.exists("./data/"):
    os.makedirs("./data/")

from gui.application import Application
from brain.core import Core


def main():
    # Redirect stdout/stderr to file
    with open("./data/run.log", "a+") as log_fh:
        # sys.stdout = log_fh
        # sys.stderr = log_fh
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        seper = "-" * 19
        print("\n\n%s\n%s\n%s\n" % (seper, now, seper))
        print('不要在此窗口进行任何操作，此窗口用于查看程序运行状态及错误分析\n请通过可视化交互界面设置实验参数或启动图像分析', end='\n\n')

        # 创建GUI应用程序和brain核心。
        app = Application()
        core = Core()

        # 确保应用程序和核心有一个句柄。
        app.set_core(core)
        core.set_gui(app)

        # 启动大脑核心运行。
        core.start()

        # core.initialise_gui()

        # 运行应用程序。
        app.mainloop()


if __name__ == "__main__":
    main()
