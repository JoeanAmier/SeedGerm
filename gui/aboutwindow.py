# -*- coding: utf-8 -*-

import tkinter as Tkinter

about_text = """
SeedGerm - Beta release

隶属: Crop Phenomics Group, Earlham Institute, Norwich Research Park, UK

作者:  Joshua Colmer, Aaron Bostrom, Ji Zhou, Danny Websdale, Thomas Le Cornu, Joshua Ball
"""


class AboutWindow(Tkinter.Toplevel):
    """
    关于窗口
    """

    def __init__(self, exp):
        Tkinter.Toplevel.__init__(self)
        self.title("关于 SeedGerm")  # 标题栏
        self.resizable(width=False, height=False)  # 不允许更改窗口大小
        self.wm_geometry("420x250")  # 窗口大小
        self.iconbitmap('.\logo.ico')  # 窗口图标

        photo = Tkinter.PhotoImage(file="./icon.gif")  # 背景图片
        w = Tkinter.Label(self, image=photo)
        w.photo = photo
        w.pack()  # 类似输出

        self.msg = Tkinter.Message(self, text=about_text)
        self.msg.pack()
