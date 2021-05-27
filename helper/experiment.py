import os
import shutil

from tinydb import Query
from tinydb import TinyDB


class Experiment(object):
    db_path = './data/experiment_list.json'  # 储存全部实验数据
    sub_directories = ['masks', 'gzdata', 'results', 'images']

    database = TinyDB(db_path)
    query = Query()
    updated = False

    def __init__(self, name=None, exp_path=None, img_path=None, panel_n=None, seeds_col_n=None, seeds_row_n=None,
                 species="Brassica", start_img=0, end_img=None, bg_remover="GMM", panel_labelled=False,
                 _yuv_ranges_set=False,
                 _eid=None, _status="", use_colour=False, use_delta=False):
        self.name = name  # 实验名称
        self.img_path = img_path  # 图像路径
        self.panel_n = panel_n  # 面板数
        self.seeds_col_n = seeds_col_n  # 列数
        self.seeds_row_n = seeds_row_n  # 行数
        self.species = species  # 品种
        self.start_img = start_img  # 起始图片
        self.end_img = end_img  # 结束图片
        self.bg_remover = bg_remover  # 背景算法
        self.panel_labelled = panel_labelled  # 参数不明
        self._yuv_ranges_set = _yuv_ranges_set  # 设置YUV
        self._eid = _eid  # 实验标识
        self._status = _status  # 状态
        self.use_colour = use_colour  # 使用颜色特征
        self.use_delta = use_delta  # 使用增量要素

        self.exp_path = exp_path  # "./data/experiments/%s" % (slugify(self.name))

        if self.exp_path is not None:
            self.create_directories()

    @property
    def seeds_n(self):
        return self.seeds_col_n * self.seeds_row_n

    @property
    def status(self):
        return self._status

    @property
    def yuv_ranges_set(self):
        return self._yuv_ranges_set

    @property
    def eid(self):
        return self._eid

    # for some reason tinyDB is indexed from 1....

    @yuv_ranges_set.setter
    def yuv_ranges_set(self, value):
        if self._yuv_ranges_set is not value:
            self._yuv_ranges_set = value
            # self.database.update(vars(self), eids=[self.eid])
            self.database.update(vars(self), self.query._eid == self.eid)
            Experiment.updated = True

    @status.setter
    def status(self, value):
        if self._status is not value:
            self._status = value
            # self.database.update(vars(self), eids=[self.eid])
            self.database.update(vars(self), self.query._eid == self.eid)
            Experiment.updated = True

    @eid.setter
    def eid(self, value):
        if self._eid is not value:
            self._eid = value  # 获取实验标识
            # self.database.update(vars(self), eids=[self.eid])
            self.database.update(vars(self), self.query.name == self.name)
            # 不需要更新GUI

    def get_results_dir(self):
        return os.path.join(self.exp_path, Experiment.sub_directories[2])

    def get_images_dir(self):
        return os.path.join(self.exp_path, Experiment.sub_directories[3])

    def get_masks_dir(self):
        return os.path.join(self.exp_path, Experiment.sub_directories[0])

    def get_results_graph(self):
        return os.path.join(self.get_results_dir(), "results.jpg")

    def create_directories(self):
        """创建实验文件夹及其子文件夹"""
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
            for sub_dir in Experiment.sub_directories:
                os.makedirs(os.path.join(self.exp_path, sub_dir))

    def reset(self):
        # 删除树并重建目录。
        if os.path.exists(self.exp_path):
            shutil.rmtree(self.exp_path)
            self.create_directories()

    def insert_into_database(self):
        # 保存数据到json
        self.eid = Experiment.database.insert(vars(self))
