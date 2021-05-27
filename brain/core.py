# -*- coding: utf-8 -*-

""" core.py - 运行一个线程，以便沿着GUI执行繁重的处理。

控制用于发芽实验的图像处理线程的启动。
进一步用于执行任何繁重的处理，以便在用户与应用程序交互时GUI线程不会挂起。
"""

import glob
import json
import os
import threading
import time
import zipfile

from brain.processor import ImageProcessor
from brain.speciesclassifier import SpeciesClassifier


class Core(threading.Thread):

    def __init__(self):
        super(Core, self).__init__()
        # threading.Thread.__init__(self)
        self.running = True
        self.current_experiment_threads = {}
        self._load_config_json()

    def _load_config_json(self):
        data = json.load(open('config.json'))
        self.chunk_no = data["chunk_no"]
        self.chunk_reverse = data["chunk_reverse"]
        self.proportions = data["proportions"]

        species_list = data["seeds"]
        self.species_classes = {}

        print('种子品种列表：')
        for species in species_list:
            obj = SpeciesClassifier(**species)  # 从字典创建对象
            self.species_classes[obj.seed] = obj
            print(obj.seed)
        print('')

    def run(self):
        """ 不是一个特别好的阻止方式。。。但要保持线的活力。"""
        while self.running:
            time.sleep(0.5)

    def set_gui(self, gui):
        """ 设置gui应用程序的句柄。 """
        self.gui = gui

    def die(self):
        """ 处理此线程和所有子线程的停止。 """
        self.running = False

        for ip in self.current_experiment_threads.values():
            ip.running = False

    def stop_processor(self, eid):
        if eid not in self.current_experiment_threads.keys():
            return

        if self.current_experiment_threads[eid].running:
            return

        # 如果我们到了这里，那么实验已经在进行中，但不再运行。
        del self.current_experiment_threads[eid]

    def start_processor(self, exp):
        """ 开始图像处理实验。 """
        if exp.eid not in self.current_experiment_threads.keys():
            self.current_experiment_threads[exp.eid] = ImageProcessor(self, self.gui, exp)
            self.current_experiment_threads[exp.eid].start()
        else:
            if not self.current_experiment_threads[exp.eid].running:
                self.stop_processor(exp.eid)
            else:
                print("该实验已经开始分析")

    def zip_results(self, exp, out_dir):
        print(exp.name)
        print(exp.exp_path)
        name_slug = os.path.basename(exp.exp_path)
        zip_f_name = "%s_results.zip" % name_slug
        out_f = os.path.join(out_dir, zip_f_name)
        print(out_f)

        exp_results_dir = exp.get_results_dir()
        to_zip = glob.glob(os.path.join(exp_results_dir, "*.csv"))
        to_zip.append(os.path.join(exp_results_dir, "results.jpg"))

        to_zip += glob.glob(os.path.join(exp.get_images_dir(), "*"))

        zip_fh = zipfile.ZipFile(out_f, "w")
        for f_name in to_zip:
            zip_fh.write(f_name, os.path.basename(f_name))
        zip_fh.close()
