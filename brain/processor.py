# -*- coding: utf-8 -*-

""" processor.py - 主要处理发芽实验。

设置用于读取和写入数据的实验目录变量。包括用于处理每个主要处理部分的数据的各种函数。所有阶段都会生成数据，以检查流程是否正常运行。

实验过程如下：
a.  需要用户设置YUV范围。
1.  保存初始图像。
2.  提取面板并保存数据。
3.  保存显示提取的面板边界的等高线图像。
4.  提取bg/fg像素并训练集成分类器。
5.  移除为每个图像生成fg遮罩的背景。
6.  使用前几帧标记种子位置。
7.  进行发芽分类。
8.  分析结果产生所需的输出数据。
9.  对种子形态和颜色数据进行量化。
"""

import copy
import glob
import json
# General python imports.
from tqdm import tqdm
import pandas as pd
import pickle
import threading
import traceback
from helper.experiment import Experiment
from helper.functions import *
# Germapp imports.
from helper.horprasert import *
from helper.panel_segmenter import fill_border
from helper.panelclass import Panel
from helper.seedpanelclass import SeedPanel
from itertools import chain
from matplotlib import pyplot as plt
from numpy import random
from operator import itemgetter
# Imaging/vision imports.
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# 机器学习和统计。
from skimage.morphology import *
from skimage.segmentation import clear_border
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from imageio import imread
import warnings

warnings.filterwarnings("ignore")

# 能够重现结果
np.random.seed(0)
random.seed(0)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_crop_shape(target, refer):
    """获得作物形状"""
    # 宽度，第三维
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # 高度，第二维
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


def create_unet(img_shape, num_class):
    # 定义并返回U-Net模型，例如层数、过滤器数、激活函数等。

    concat_axis = 3
    inputs = layers.Input(shape=img_shape)

    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
    up6 = layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = layers.Cropping2D(cropping=(ch, cw))(conv3)
    up7 = layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
    up8 = layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = layers.Cropping2D(cropping=(ch, cw))(conv1)
    up9 = layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
    conv10 = layers.Conv2D(num_class, (1, 1), activation='softmax')(conv9)

    return models.Model(inputs=inputs, outputs=conv10)


def compare_pts(pt1, pt2):
    return pt1 > pt2


class ImageProcessor(threading.Thread):
    def __init__(self, core=None, app=None, experiment=None):
        """ 设置并生成所需的实验数据变量。 """
        super(ImageProcessor, self).__init__()

        self.core = core
        self.app = app
        self.exp = experiment  # type: Experiment
        self.running = True

        self.all_masks = []
        self.total_stats = []

        # 读取实验图像文件名，按图像编号排序
        self.imgs = get_images_from_dir(self.exp.img_path)
        self.all_imgs = [pj(self.exp.img_path, el) for el in self.imgs]

        self.exp_masks_dir = pj(self.exp.exp_path, "masks")
        self.exp_masks_dir_frame = pj(self.exp_masks_dir, "frame_%d.npy")
        self.exp_images_dir = pj(self.exp.exp_path, "images")
        self.exp_results_dir = pj(self.exp.exp_path, "results")
        self.exp_gzdata_dir = pj(self.exp.exp_path, "gzdata")

        self.rprops = []
        self.all_rprops = []
        self.all_imgs_list = []
        for image_path in self.all_imgs:
            im = imread(image_path)
            self.all_imgs_list.append(im)

        self.yuv_json_file = os.path.join(
            self.exp.exp_path,
            "yuv_ranges.json"
        )

        with open(self.yuv_json_file) as fh:
            data = json.load(fh)

        self.yuv_low = np.array(data['low'])
        self.yuv_high = np.array(data['high'])

        try:
            self.spp_processor = copy.deepcopy(self.core.species_classes[
                                                   self.exp.species])
        except KeyError:
            # print("No species module found for %s" % (self.exp.species))
            # print("ought to use default, shouldn't occur as populate species list from these modules")
            # print("consider adding parameters to the config if you're confident")
            print("无效的 %s 品种参数" % (self.exp.species))
            print("应该选择已有的品种，不应该使用品种列表以外的品种")
            print("如果您有信心，可以考虑在配置中添加新的品种参数")

    def _run_check(self):
        pass

    def _save_init_image(self, img):
        # 保存实验的初始RGB图像
        out_f = pj(self.exp_images_dir, "init_img.jpg")
        if os.path.exists(out_f):
            return
        img_01 = self.all_imgs_list[0] / 255.
        fig = plt.figure(dpi=600)
        plt.imshow(img_01)
        fig.savefig(out_f)
        plt.close(fig)

    def _yuv_clip_image(self, img_f):
        # 使用手动设置的范围返回图像的二进制遮罩
        img = self.all_imgs_list[img_f]
        img_yuv = rgb2ycrcb(img)
        mask_img = in_range(img_yuv, self.yuv_low, self.yuv_high)
        return mask_img.astype(np.bool)

    def _yuv_clip_panel_image(self, img_f, p):
        # 使用单个面板范围返回面板图像的二进制遮罩
        img = self.all_imgs_list[img_f]
        img_yuv = rgb2ycrcb(img)
        self.yuv_panel_json_file = os.path.join(
            self.exp.exp_path,
            "yuv_ranges_{}.json".format(p)
        )

        if os.path.exists(self.yuv_panel_json_file):
            with open(self.yuv_panel_json_file) as fh:
                data = json.load(fh)
        else:
            with open(self.yuv_json_file) as fh:
                data = json.load(fh)

        self.yuv_panel_low = np.array(data['low'])
        self.yuv_panel_high = np.array(data['high'])
        mask_img = in_range(img_yuv, self.yuv_panel_low, self.yuv_panel_high)
        return mask_img.astype(np.bool)

    def _extract_panels(self, img, chunk_no, chunk_reverse, img_idx):
        # 如果面板数据已经存在，请加载它
        panel_data_f = os.path.join(self.exp_gzdata_dir, "panel_data.pkl")
        if os.path.exists(panel_data_f):
            with open(panel_data_f, 'rb') as fh:
                try:
                    self.panel_list = pickle.load(fh)
                    return
                except EOFError:
                    print("序列化数据已损坏")

        # 使用设置的YUV范围获取二进制遮罩
        mask_img = self._yuv_clip_image(img_idx)
        mask_img = remove_small_objects(
            fill_border(mask_img, 10, fillval=False),
            min_size=1024
        )
        mask_img_cleaned_copy = mask_img.copy()
        mask_img = erosion(binary_fill_holes(mask_img), disk(7))
        obj_only_mask = np.logical_and(mask_img, np.logical_not(mask_img_cleaned_copy))

        # 排序遮罩中的面板，计算遮罩中找到的面板数
        ordered_mask_img, n_panels = measurements.label(mask_img)
        # 返回列表中每个面板的区域属性
        rprops = regionprops(ordered_mask_img, coordinates='xy')
        # 移除不太可能是真实面板的项目
        rprops = [x for x in rprops if
                  x.area > self.all_imgs_list[0].shape[0] * self.all_imgs_list[0].shape[1] / self.exp.panel_n / 6]

        def get_mask_objects(idx, rp):
            tmp_mask = np.zeros(mask_img.shape)
            tmp_mask[ordered_mask_img == rp.label] = 1

            both_mask = np.logical_and(obj_only_mask, tmp_mask)
            both_mask = remove_small_objects(both_mask)
            _, panel_object_count = measurements.label(both_mask)  # , return_num=True)
            return panel_object_count, tmp_mask, both_mask

        rprops = [(rp, get_mask_objects(idx, rp)) for idx, rp in enumerate(rprops)]
        rprops = [[item[0], item[1][0], item[1][1], item[1][2]] for idx, item in enumerate(rprops)]
        rprops = sorted(rprops, key=itemgetter(1), reverse=True)

        # 检查面板是否有种子
        panels = [(rp, rp.centroid[0], rp.centroid[1], tmp, both) for rp, _, tmp, both in rprops[:self.exp.panel_n]]

        # 先按y排序，然后按x排序
        panels = sorted(panels, key=itemgetter(1))
        panels = chunks(panels, chunk_no)
        panels = [sorted(p, key=itemgetter(2), reverse=chunk_reverse) for p in panels]
        panels = list(chain(*panels))

        # 设置遮罩，其中1为左上角，2为右上角，3为左中角，依此类推
        panel_list = []
        tmp_list = []
        both_list = []
        for idx in range(len(panels)):
            rp, _, _, tmp, both = panels[idx]
            new_mask = np.zeros(mask_img.shape)
            new_mask[ordered_mask_img == rp.label] = 1
            panel_list.append(Panel(idx + 1, new_mask.astype(np.bool), rp.centroid, rp.bbox))
            tmp_list.append(tmp)
            both_list.append(both)

        self.panel_list = panel_list
        self.rprops = rprops
        for i in range(len(tmp_list)):
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Mask for panel {}'.format(i + 1))
            ax[0].imshow(tmp_list[i])
            ax[1].imshow(both_list[i])
            fig.savefig(pj(self.exp_images_dir, "mask_img_{}.jpg".format(i + 1)))
            plt.close(fig)

        with open(panel_data_f, "wb") as fh:
            pickle.dump(self.panel_list, fh)

    def _save_contour_image(self):
        fig = plt.figure(dpi=600)
        img_01 = self.all_imgs_list[self.exp.start_img]
        # plt.imshow(img_01)

        img_full_mask = np.zeros(img_01.shape[:-1])
        for p in self.panel_list:
            plt.annotate(str(p.label), xy=p.centroid[::-1], color='r', fontsize=20)
            min_row, min_col, max_row, max_col = p.bbox
            img_full_mask[min_row:max_row, min_col:max_col] += p.mask_crop

        self.panels_mask = img_full_mask.astype(np.bool)

        out_f = pj(self.exp_images_dir, "img_panels.jpg")
        if os.path.exists(out_f):
            return

        plt.gca().invert_yaxis()
        plt.contour(img_full_mask, [0.5], colors='r')
        fig.savefig(out_f)
        plt.close(fig)

        fig = plt.figure(dpi=600)
        plt.imshow(self.panels_mask)
        fig.savefig(pj(self.exp_images_dir, "panels_mask.jpg"))
        plt.close(fig)

    def _gmm_update_gmm(self, gmm, y):
        y_pred = gmm.predict(y)

        # Previously used 1 / # examples, but parameters are updated too quickly
        alpha = 0.25 / float(y.shape[0])

        for x, m in zip(y, y_pred):
            obs = np.zeros(gmm.n_components)
            obs[m] = 1.
            delta_m = x - gmm.means_[m]
            new_weights = gmm.weights_ + alpha * (obs - gmm.weights_)
            new_mean = gmm.means_[m] + (alpha / gmm.means_[m]) * delta_m
            new_std = gmm.covariances_[m] + (alpha / gmm.means_[m]) * (
                    delta_m.T * delta_m - np.power(gmm.covariances_[m], 2))
            gmm.weights_ = new_weights
            gmm.means_[m] = new_mean
            gmm.covariances_[m] = new_std

    def _gmm_get_TCD(self, X, E, s, b):
        A = np.sum((X * E) / np.power(s, 2), axis=1)
        B = np.sum(np.power(E / s, 2))
        alpha = A / B

        alpha_tiled = np.repeat(alpha.reshape(alpha.shape[0], 1), 3, axis=1)
        inner = np.power((X - (E * alpha_tiled)) / s, 2)
        all_NCD = np.sqrt(np.sum(inner, axis=1)) / b
        return np.percentile(all_NCD, 99.75)

    def _train_gmm_clfs(self):
        gmm_clf_f = pj(self.exp_gzdata_dir, "gmm_clf.pkl")
        if os.path.exists(gmm_clf_f):
            with open(gmm_clf_f, 'rb') as fh:
                self.classifiers = pickle.load(fh)
            return

        curr_img = self.all_imgs_list[self.exp.start_img] / 255.

        curr_mask = self._yuv_clip_image(self.exp.start_img)

        curr_mask = dilation(curr_mask, disk(2))

        bg_mask3 = np.dstack([np.logical_and(curr_mask, self.panels_mask)] * 3)
        bg_rgb_pixels = curr_img * bg_mask3

        # 获取所有bg像素
        bg_rgb_pixels = flatten_img(bg_rgb_pixels)
        bg_rgb_pixels = bg_rgb_pixels[bg_rgb_pixels.all(axis=1), :]
        bg_retain = int(bg_rgb_pixels.shape[0] * 0.1)
        bg_retain = random.choice(bg_rgb_pixels.shape[0], bg_retain, replace=True)
        X = bg_rgb_pixels[bg_retain, :]

        blue_E, blue_s = X.mean(axis=0), X.std(axis=0)

        alpha = flatBD(X, blue_E, blue_s)
        a = np.sqrt(np.power(alpha - 1, 2) / X.shape[0])
        b = np.sqrt(np.power(flatCD(X, blue_E, blue_s), 2) / X.shape[0])

        TCD = self._gmm_get_TCD(X, blue_E, blue_s, b)
        # print("Training GMM background remover...")
        print("正在训练GMM背景移除")
        bg_gmm = GaussianMixture(n_components=3, random_state=0)
        bg_gmm.fit(X)

        thresh = np.percentile(bg_gmm.score(X), 1.)

        new_E = (bg_gmm.means_ * bg_gmm.weights_.reshape(1, bg_gmm.n_components).T).sum(axis=0)

        self.classifiers = [bg_gmm, blue_E, blue_s, TCD, thresh, a, b, new_E]

        with open(gmm_clf_f, "wb") as fh:
            pickle.dump(self.classifiers, fh)

    def _gmm_remove_background(self):
        if len(os.listdir(self.exp_masks_dir)) >= ((self.exp.end_img - self.exp.start_img) - self.exp.start_img):
            return

        bg_gmm, blue_E, blue_s, TCD, thresh, a, b, new_E = self.classifiers

        if len(os.listdir(self.exp_masks_dir)) >= (self.exp.end_img - self.exp.start_img):
            return

        # 2d array of all the masks that are indexed first by image, then by panel.
        self.all_masks = []
        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img = self.all_imgs_list[idx] / 255.

            img_masks = []
            # 为每个面板生成预测的遮罩，并将它们添加到遮罩列表中。
            for p in self.panel_list:
                panel_img = p.get_cropped_image(img)
                pp_predicted = NCD(panel_img, new_E, blue_s, b) > TCD
                pp_predicted = pp_predicted.astype(np.bool)
                img_masks.append(pp_predicted)

            y = flatten_img(img)
            predicted = bg_gmm.score_samples(y)
            if new_E[0] < 1e-4:
                with open(self.exp_masks_dir_frame % (idx), "wb") as fh:
                    np.save(fh, img_masks)
                self.all_masks.append(img_masks)
                continue
            bg_retain = predicted > thresh
            y_bg = y[bg_retain, :]
            retain = random.choice(y_bg.shape[0], min(y_bg.shape[0], 100000), replace=False)
            y_bg = y_bg[retain, :]
            self._gmm_update_gmm(bg_gmm, y_bg)

            new_E = (bg_gmm.means_ * bg_gmm.weights_.reshape(1, bg_gmm.n_components).T).sum(axis=0)

            print(new_E)

            with open(self.exp_masks_dir_frame % (idx), "wb") as fh:
                np.save(fh, img_masks)

            self.all_masks.append(img_masks)

    def _ensemble_predict(self, clfs, X, p):
        return clfs[p.label - 1].predict(X)

    def _train_clfs(self, clf_in):
        print("分类器：", self.exp.bg_remover)
        self.classifiers = []
        # 对于面板列表中的每个面板，创建训练数据，然后训练定义的分类器
        for p in self.panel_list:
            print("第 {} 个面板分类器开始训练".format(p.label))

            # 如果分类器已经存在，尝试加载它们
            ensemble_clf_f = pj(self.exp_gzdata_dir, "ensemble_clf_{}.pkl".format(p.label))
            if os.path.exists(ensemble_clf_f):
                with open(ensemble_clf_f, 'rb') as fh:
                    self.classifiers = pickle.load(fh)

            if len(self.classifiers) == len(self.panel_list):
                print('已成功加载已有的分类器')
                return

            # 4 x 4图定义，每个子图将显示一个训练遮罩
            fig, axarr = plt.subplots(4, 4)
            fig.suptitle('第 {}  个面板的图像训练'.format(p.label))
            axarr = list(chain(*axarr))

            train_masks = []
            train_images = []

            # 选择16个训练图像作为前10个ID、中间ID和最后5个ID
            train_img_ids = list(range(self.exp.start_img, self.exp.start_img + 10))
            train_img_ids += [int((self.exp.end_img + self.exp.start_img) / 2) - 1, self.exp.end_img - 2,
                              self.exp.end_img - 1, self.exp.end_img - 3, self.exp.end_img - 4, self.exp.end_img - 5]

            # 对于每个训练图像，使用在实验设置中选择的yuv阈值创建训练遮罩标签
            # 在4 x 4图中绘制的训练遮罩标签
            for idx, img_i in enumerate(train_img_ids):
                curr_img = self.all_imgs_list[img_i][p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] / 255.
                train_images.append(curr_img)

                curr_mask = self._yuv_clip_panel_image(img_i, p.label)[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
                curr_mask = dilation(curr_mask, disk(2))
                train_masks.append(curr_mask.astype(np.bool))

                axarr[idx].imshow(curr_mask)
                axarr[idx].axis('off')

            # 此图显示了保存在images目录中的此面板的16个训练图像
            fig.savefig(pj(self.exp_images_dir, "train_imgs_panel_{}.jpg".format(p.label)))
            plt.close(fig)

            all_bg_pixels = []
            all_fg_pixels = []

            # 对于每个训练遮罩，获取背景和前景像素的相应RGB值，并附加到前景位置的所有RGB值列表以及背景位置的所有RGB值列表
            for idx, (mask, curr_img) in enumerate(zip(train_masks, train_images)):
                bg_mask3 = np.dstack(
                    [np.logical_and(mask, self.panels_mask[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]])] * 3)
                fg_mask3 = np.dstack([np.logical_and(np.logical_not(mask),
                                                     self.panels_mask[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]])] * 3)

                bg_rgb_pixels = self._create_transformed_data(
                    curr_img * bg_mask3)
                fg_rgb_pixels = self._create_transformed_data(
                    curr_img * fg_mask3)

                all_bg_pixels.append(bg_rgb_pixels)
                all_fg_pixels.append(fg_rgb_pixels)

            bg_rgb_pixels = np.vstack(all_bg_pixels)
            fg_rgb_pixels = np.vstack(all_fg_pixels)

            # 连接来自所有图像的训练数据，X包含对应于Y标签值的RGB值（0为背景，1为前景）
            X = np.vstack([bg_rgb_pixels, fg_rgb_pixels])
            y = np.concatenate([
                np.zeros(bg_rgb_pixels.shape[0]),
                np.ones(fg_rgb_pixels.shape[0])
            ])

            # 在此面板的训练数据上训练分类器
            self._train_clf(clf_in, ensemble_clf_f, X, y)

    def _train_unet(self):
        print("分类器：", self.exp.bg_remover)

        callbacks = [
            # EarlyStopping(patience=20, verbose=1),
            # ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
            # ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
        ]

        images = self.all_imgs_list

        print("选择训练数据")

        train_img_ids = list(range(self.exp.start_img, self.exp.start_img + 10))
        train_img_ids += [int((self.exp.end_img + self.exp.start_img) / 2) - 1,
                          int((self.exp.end_img + self.exp.start_img) / 2) - 2,
                          int((self.exp.end_img + self.exp.start_img) / 2) - 3,
                          int((self.exp.end_img + self.exp.start_img) / 2) - 4, self.exp.end_img - 2,
                          self.exp.end_img - 1, self.exp.end_img - 3, self.exp.end_img - 4]

        train_images = []
        train_masks = []

        for idx, img_i in enumerate(train_img_ids):
            curr_img = self.all_imgs_list[img_i] / 255.
            curr_img = resize(curr_img, (int(curr_img.shape[0] / 2), int(curr_img.shape[1] / 2)),
                              anti_aliasing=True)
            curr_img = np.expand_dims(curr_img, axis=0)
            train_images.append(curr_img)

            curr_mask = self._yuv_clip_image(img_i)
            curr_mask = dilation(curr_mask, disk(2))
            curr_mask = resize(curr_mask, (int(curr_mask.shape[0] / 2), int(curr_mask.shape[1] / 2)),
                               anti_aliasing=True)
            curr_mask = to_categorical(curr_mask, num_classes=2, dtype='uint8')
            curr_mask = np.expand_dims(curr_mask, axis=0)
            train_masks.append(curr_mask.astype(np.bool))

        panel_images = {}
        panel_masks = {}

        for p in self.panel_list:
            panels_img = []
            panel_mask = []
            for idx, img_i in enumerate(train_img_ids):
                img = self.all_imgs_list[idx]
                panel_img = p.get_cropped_image(img)
                panel_img = np.expand_dims(panel_img, axis=0)
                panels_img.append(panel_img)
                curr_mask = self._yuv_clip_image(img_i)
                curr_mask = dilation(curr_mask, disk(2))
                curr_mask = to_categorical(curr_mask, num_classes=2, dtype='uint8')
                curr_mask = p.get_cropped_image(curr_mask)
                curr_mask = np.expand_dims(curr_mask, axis=0)
                panel_mask.append(curr_mask)
            panel_images[p.label] = panels_img
            panel_masks[p.label] = panel_mask

        X = np.vstack(train_images)
        Y = np.vstack(train_masks)

        print("所选训练数据：", X.shape, Y.shape)

        models = {}

        for j in range(1, len(panel_images) + 1):
            panel_images_j = panel_images[j]
            panel_images_j = np.vstack(panel_images_j)
            panel_masks_j = panel_masks[j]
            panel_masks_j = np.vstack(panel_masks_j)
            pshape = panel_images_j.shape[1:]
            model = create_unet(self, pshape, 2)
            callbacks = [
                EarlyStopping(patience=20, verbose=1),
                ReduceLROnPlateau(factor=0.3, patience=10, min_lr=0.0000001, verbose=1),
                ModelCheckpoint('model_panel%s.h5' % j, verbose=1, save_best_only=True, save_weights_only=False)
            ]
            model.compile(optimizer=Adam(lr=0.00001), loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(panel_images_j, panel_masks_j, batch_size=2, epochs=200, callbacks=callbacks, verbose=2,
                      validation_split=0.2)
            models[j] = load_model('model_panel%s.h5' % j)

        score = models[1].predict(panel_images[1][0])
        preds = (score > 0.5).astype('uint8')
        preds1 = preds[0, :, :, :]
        preds1 = np.argmax(preds1, axis=2)

        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img_masks = []
            img = self.all_imgs_list[idx]
            for p in self.panel_list:
                panel_img = p.get_cropped_image(img)
                panel_img = np.expand_dims(panel_img, axis=0)
                pp_predicted = models[p.label].predict(panel_img)
                pp_predicted = np.argmax((pp_predicted > 0.4).astype('uint8')[0, :, :, :], axis=2)
                pp_predicted.shape = p.mask_crop.shape
                pp_predicted = pp_predicted.astype(np.bool)
                img_masks.append(pp_predicted)
            fig = plt.figure(dpi=300)
            plt.imshow(pp_predicted)
            fig.savefig(pj(self.exp_images_dir, "panels_mask_%s.jpg" % idx))  # MASKS
            plt.close(fig)
            self.all_masks.append(img_masks)

            with open(self.exp_masks_dir_frame % idx, "wb") as fh:
                np.save(fh, img_masks)

        self.app.status_string.set("正在移除背景 %d %%" % int(
            float(idx) / (float(self.exp.end_img - self.exp.start_img)) * 100))

    def _train_clf(self, clf_in, ensemble_clf_f, X, y):
        # Split X and Y into train/test to get an estimate of classification accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("完整数据集的形状：", X.shape, y.shape)
        print("训练数据集的形状：", X_train.shape, y_train.shape)
        print("测试数据集的形状：", X_test.shape, y_test.shape)

        # 在训练数据上拟合分类器，打印训练和测试精度分数
        for clf_n, clf in clf_in:
            clf.fit(X_train, y_train)
            print(clf_n, "训练得分", clf.score(X_train, y_train))
            print(clf_n, "检验得分：", clf.score(X_test, y_test))
        # 将训练好的分类器附加到分类器列表中
        self.classifiers.append(clf)

        # 保存训练分类器
        with open(ensemble_clf_f, "wb") as fh:
            pickle.dump(clf_in, fh)

    # def _sgd_hyperparameters(self, clf_in, ensemble_clf_f, X, y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #     params = {'alpha': [0.0003], 'max_iter': [50000], 'tol': [1e-4], 'random_state': [0],
    #               'learning_rate': ['optimal'], 'early_stopping': [True], 'validation_fraction': [0.2],
    #               'n_iter_no_change': [4], 'average': [True], 'n_jobs': [-1]}
    #     clf = SGDClassifier()
    #     grid = GridSearchCV(clf, param_grid=params, cv=5, refit=True, verbose=2)
    #     print(X.shape)
    #     print(y.shape)
    #     grid.fit(X, y)
    #     print("Best validation score: ", grid.best_score_)
    #     print("Best validation params: ", grid.best_params_)
    #     y_pred = grid.predict(X_test)
    #     print("Score of the optimised classifiers: ", accuracy_score(y_test, y_pred))
    #     grid = grid.best_estimator_
    #     with open(ensemble_clf_f, "wb") as fh:
    #         pickle.dump(grid, fh)

    def _create_transformed_data(self, rgb_pixels):
        # 删除随机选择，以便尽可能多地使用数据。
        # 获取所有bg像素
        rgb_pixels = flatten_img(rgb_pixels)
        rgb_pixels = rgb_pixels[rgb_pixels.all(axis=1), :]
        return rgb_pixels

    def _remove_background(self):
        print("正在移除背景")

        # 如果遮罩目录中的遮罩数大于要分析的图像数，请跳过背景删除步骤，因为遮罩已经存在
        if len(os.listdir(self.exp_masks_dir)) >= (self.exp.end_img - self.exp.start_img):
            return

        self.all_masks = []

        # 预测每个图像中的背景和前景像素
        for idx in tqdm(range(self.exp.start_img, self.exp.end_img)):
            img_masks = []
            # 尝试加载此图像ID的预测背景前景遮罩（如果存在）
            try:
                img_masks = np.load(self.exp_masks_dir_frame % idx, allow_pickle=True)
            except Exception as e:
                img = self.all_imgs_list[idx]
                # 对于此图像中的每个面板，将图像裁剪为感兴趣的面板，预测此面板中的BGFG像素，对预测的遮罩应用膨胀和腐蚀
                # 移除不太可能成为种子的小对象。将此预测的遮罩附加到此图像的遮罩列表中。
                for p in self.panel_list:
                    panel_img = p.get_cropped_image(img)
                    pp_predicted = self._ensemble_predict(self.classifiers, flatten_img(panel_img), p)
                    pp_predicted.shape = p.mask_crop.shape
                    pp_predicted = pp_predicted.astype(np.bool)
                    # pp_predicted = dilation(pp_predicted, disk(4))
                    # pp_predicted = erosion(pp_predicted, disk(2))
                    pp_predicted = remove_small_objects(pp_predicted)
                    img_masks.append(pp_predicted)

                # 保存此图像的遮罩列表
                with open(self.exp_masks_dir_frame % idx, "wb") as fh:
                    np.save(fh, img_masks)

            # 将此图像的遮罩列表附加到包含所有图像遮罩的列表
            self.app.status_string.set("正在移除背景 %d %%" % int(
                float(idx) / (float(self.exp.end_img - self.exp.start_img)) * 100))
            self.all_masks.append(img_masks)

    def _label_seeds(self):
        l_rprops_f = pj(self.exp_gzdata_dir, "l_rprops_data.pkl")
        if os.path.exists(l_rprops_f):
            with open(l_rprops_f, 'rb') as fh:
                try:
                    self.panel_l_rprops = pickle.load(fh)
                    return
                except EOFError:
                    print("读取序列化数据失败")

        fig, axarr = plt.subplots(self.exp.panel_n, 1, figsize=(16, 16 * self.exp.panel_n))
        try:
            axarr.shape
        except:
            axarr = [axarr]

        retain = int((self.exp.end_img - self.exp.start_img) / 10.)

        init_masks = self.all_masks

        if len(init_masks) == 0:
            for i in range(self.exp.start_img, self.exp.start_img + retain):
                data = np.load(self.exp_masks_dir_frame % i, allow_pickle=True)
                init_masks.append(data)
        self.panel_l_rprops = []

        for idx, panel in enumerate(self.panel_list):

            # :10 for tomato, :20 for corn/brassica
            mask_med = np.dstack([img_mask[idx] for img_mask in init_masks])
            mask_med = clear_border(np.median(mask_med, axis=2)).astype(np.bool)
            mask_med = remove_small_objects(mask_med)

            # 使用默认结构元素（十字）标记数组中的特征。
            labelled_array, num_features = measurements.label(mask_med)
            rprops = regionprops(labelled_array, coordinates='xy')

            all_seed_rprops = [
                SeedPanel(
                    rp.label,
                    rp.centroid,
                    rp.bbox,
                    rp.moments_hu,
                    rp.area,
                    rp.perimeter,
                    rp.eccentricity,
                    rp.major_axis_length,
                    rp.minor_axis_length,
                    rp.solidity,
                    rp.extent,
                    rp.convex_area,
                )
                for rp in rprops
            ]

            # 获取最大种子数
            pts = np.vstack([el.centroid for el in all_seed_rprops])
            in_mask = find_closest_n_points(pts, self.exp.seeds_n)

            # 如果我们得到的种子比我们应该得到的少，我们应该扔掉它们吗？
            if len(in_mask) > self.exp.seeds_n:
                all_seed_rprops_new = []
                for rp, im in zip(all_seed_rprops, in_mask):
                    if im:
                        all_seed_rprops_new.append(rp)
                    else:
                        # 从遮罩中删除假种子rprops，之后可能需要重新排序
                        labelled_array[labelled_array == rp.label] = 0
                all_seed_rprops = all_seed_rprops_new
            # end if-----------------------------------#

            # 从边界移除额外的“种子”（QR标签）
            pts = np.vstack([el.centroid for el in all_seed_rprops])
            xy_range = get_xy_range(labelled_array)
            in_mask = find_pts_in_range(pts, xy_range)

            # 如果我们的种子太多了，我们应该扔掉吗？
            if len(in_mask) > self.exp.seeds_n:
                all_seed_rprops_new = []
                for rp, im in zip(all_seed_rprops, in_mask):
                    if im:
                        all_seed_rprops_new.append(rp)
                    else:
                        # 从遮罩中删除错误的种子操作
                        labelled_array[labelled_array == rp.label] = 0
                all_seed_rprops = all_seed_rprops_new
            # end if-----------------------------------#

            # 如果我们删减了，需要更新pts。
            pts = np.vstack([el.centroid for el in all_seed_rprops])

            pts_order = order_pts_lr_tb(pts, self.exp.seeds_n, xy_range, self.exp.seeds_col_n, self.exp.seeds_row_n)

            new_order = []
            new_mask = np.zeros(labelled_array.shape)
            for s_idx, s in enumerate(pts_order):
                sr = all_seed_rprops[s]
                # 重排序遮罩
                new_mask[labelled_array == sr.label] = s_idx + 1
                sr.label = s_idx + 1
                new_order.append(sr)

            all_seed_rprops = new_order
            labelled_array = new_mask

            # 我们为每个面板添加一个标签数组和区域属性。
            self.panel_l_rprops.append((labelled_array, all_seed_rprops))

        minimum_areas = []
        self.end_idx = np.full(len(self.panel_list), self.exp.end_img)

        for idx in tqdm(range(len(self.all_masks))):
            self.panel_l_rprops_1 = []
            fig, axarr = plt.subplots(self.exp.panel_n, 1, figsize=(16, 16 * self.exp.panel_n))
            for ipx, panel in enumerate(self.panel_list):
                # :10 for tomato, :20 for corn/brassica
                # mask_med = np.dstack([img_mask[idx] for img_mask in init_masks])
                mask_med = init_masks[idx][ipx]
                mask_med = remove_small_objects(mask_med)

                # 使用默认结构元素（十字）标记数组中的特征。
                labelled_array, num_features = measurements.label(mask_med)
                rprops = regionprops(labelled_array, coordinates='xy')

                all_seed_rprops = []  # type: List[SeedPanel]
                for rp in rprops:
                    all_seed_rprops.append(
                        SeedPanel(rp.label, rp.centroid, rp.bbox, rp.moments_hu, rp.area, rp.perimeter, rp.eccentricity,
                                  rp.major_axis_length, rp.minor_axis_length, rp.solidity, rp.extent, rp.convex_area))

                if idx == 0:
                    minimum_areas.append(np.zeros((len(all_seed_rprops))))
                    for i in range(len(all_seed_rprops)):
                        minimum_areas[ipx][i] = all_seed_rprops[i].area

                # 获取最大种子数
                if all_seed_rprops == []:
                    break
                pts = np.vstack([el.centroid for el in all_seed_rprops])
                in_mask = find_closest_n_points(pts, self.exp.seeds_n)

                # 如果我们得到的种子比我们应该得到的少，我们应该扔掉它们吗？
                if len(in_mask) > self.exp.seeds_n:
                    all_seed_rprops_new = []
                    for rp, im in zip(all_seed_rprops, in_mask):
                        if rp.area < 0.6 * np.percentile(minimum_areas[ipx], 10):
                            print("Removed object with area =" + str(rp.area))
                            labelled_array[labelled_array == rp.label] = 0
                            labelled_array[labelled_array > rp.label] -= 1
                        elif im:
                            all_seed_rprops_new.append(rp)
                        else:
                            # 从遮罩中删除错误的种子操作
                            labelled_array[labelled_array == rp.label] = 0
                            labelled_array[labelled_array > rp.label] -= 1
                    all_seed_rprops = all_seed_rprops_new

                # end if-----------------------------------#

                # 从边界移除额外的“种子”（QR标签）
                pts = np.vstack([el.centroid for el in all_seed_rprops])
                xy_range = get_xy_range(labelled_array)
                in_mask = find_pts_in_range(pts, xy_range)

                # If we've got more seeds than we should do, should we throw them away?
                # if len(in_mask) > self.exp.seeds_n:
                #     all_seed_rprops_new = []
                #     for rp, im in zip(all_seed_rprops, in_mask):
                #         if im:
                #             all_seed_rprops_new.append(rp)
                #         else:
                #             # Remove false seed rprops from mask
                #             labelled_array[labelled_array == rp.label] = 0
                #     all_seed_rprops = all_seed_rprops_new
                # end if-----------------------------------#

                # 如果我们删减了，需要更新pts。
                pts = np.vstack([el.centroid for el in all_seed_rprops])

                pts_order = order_pts_lr_tb(pts, self.exp.seeds_n, xy_range, self.exp.seeds_col_n, self.exp.seeds_row_n)

                new_order = []
                new_mask = np.zeros(labelled_array.shape)
                for s_idx, s in enumerate(pts_order):
                    sr = all_seed_rprops[s]
                    # 重排序遮罩
                    new_mask[labelled_array == sr.label] = s_idx + 1
                    sr.label = s_idx + 1
                    new_order.append(sr)

                all_seed_rprops = new_order
                labelled_array = new_mask

                # 我们为每个面板添加一组标签和区域属性。
                print("面板 {} 中标识的种子数：".format(panel.label) + str(len(all_seed_rprops)))
                self.panel_l_rprops_1.append((labelled_array, all_seed_rprops))
                if self.exp.panel_n > 1:
                    axarr[ipx].imshow(mask_med)
                    for rp in all_seed_rprops:
                        axarr[ipx].annotate(str(rp.label), xy=rp.centroid[::-1] + np.array([10, -10]), color='r',
                                            fontsize=16)
                        axarr[ipx].annotate(str(panel.label), xy=(20, 10), color='r', fontsize=28)
                        axarr[ipx].axis('off')
                else:
                    axarr.imshow(mask_med)
                    for rp in all_seed_rprops:
                        axarr.annotate(str(rp.label), xy=rp.centroid[::-1] + np.array([10, -10]), color='r',
                                       fontsize=16)
                        axarr.annotate(str(panel.label), xy=(20, 10), color='r', fontsize=28)
                        axarr.axis('off')

                if len(all_seed_rprops) < 0.8 * self.exp.seeds_n and self.end_idx[ipx] == self.exp.end_img:
                    self.end_idx[ipx] = idx
            fig.savefig(pj(self.exp_images_dir, 'seeds_labelled_{}.png'.format(str(idx))))
            plt.close('all')
            self.all_rprops.append(self.panel_l_rprops_1)

    def _generate_statistics(self):
        l_rprops_f = pj(self.exp_gzdata_dir, "l_rprops_data.pkl")
        if os.path.exists(l_rprops_f):
            with open(l_rprops_f, 'rb') as fh:
                self.all_rprops = pickle.load(fh)
        for j in range(len(self.all_rprops)):
            x = self.all_rprops[j]
            n_seeds = sum(len(x[p][1]) for p in range(len(x)))
            X_stats = np.zeros((n_seeds, 10))
            counter = 0
            for i in range(len(x)):
                x0 = x[i][1]
                for k in range(len(x0)):
                    X_stats[counter, :] = [i + 1, k + 1, x0[k].area, x0[k].eccentricity, x0[k].extent,
                                           x0[k].major_axis_length, x0[k].minor_axis_length, x0[k].perimeter,
                                           x0[k].solidity, x0[k].convex_area]
                    counter += 1
            self.total_stats.append(X_stats)
        seed_stats = np.zeros((1, 11))
        for i in range(len(self.total_stats)):
            c = self.total_stats[i]
            x = np.concatenate((np.full(shape=(c.shape[0], 1), fill_value=i), c), axis=1)
            seed_stats = np.concatenate((seed_stats, x))
        seed_stats = np.delete(seed_stats, 0, axis=0)
        stats_over_time = pd.DataFrame(seed_stats, columns=['Image Index', 'Panel Number', 'Seed Number', 'Seed Area',
                                                            'Seed Eccentricity', 'Seed Extent',
                                                            'Seed Major Axis Length', 'Seed Minor Axis Length',
                                                            'Seed Perimeter', 'Seed Solidity', 'Seed Convex Area'])
        stats_over_time['Image Index'] = stats_over_time['Image Index'].astype('uint8')
        stats_over_time['Panel Number'] = stats_over_time['Panel Number'].astype('uint8')
        stats_over_time['Seed Number'] = stats_over_time['Seed Number'].astype('uint8')
        direc = pj(self.exp_results_dir, "stats_over_time.csv")
        stats_over_time.to_csv(direc, index=False)
        return

    def _perform_classification(self):
        """ 还需要量化种子是否合并，以及是否移动。"""
        print("种子分类")

        if len(glob.glob(pj(self.exp_results_dir, "germ_panel_*.json"))) >= self.exp.panel_n:
            print("数据已经分析")
            return

        if self.all_masks is None:
            self.all_masks = []
            for i in range(self.exp.start_img, self.exp.end_img):
                data = np.load(self.exp_masks_dir_frame % i, allow_pickle=True)
                self.all_masks.append(data)

        # 分别评估每个面板，这样基因型之间的差异不会恶化结果
        for panel_idx, panel_object in enumerate(tqdm(self.panel_list)):
            try:
                # 这是从regionprops生成的标签数组的元组
                panel_labels, panel_regionprops = self.panel_l_rprops[panel_idx]
                p_masks = []
                # 为每个图像提取特定面板的所有遮罩
                for i in range(len(self.all_masks)):
                    p_masks.append(self.all_masks[i][panel_idx])

                self.spp_processor.use_colour = self.exp.use_colour
                self.spp_processor.use_delta = self.exp.use_delta

                panel_germ = self.spp_processor._classify(
                    panel_object,
                    self.all_imgs[self.exp.start_img:self.exp.end_img],
                    panel_labels,
                    panel_regionprops,
                    p_masks,
                    self.all_imgs_list[self.exp.start_img:self.exp.end_img]
                )

                out_f = pj(
                    self.exp_results_dir,
                    "germ_panel_%d.json" % panel_idx
                )

                with open(out_f, "w") as fh:
                    json.dump(panel_germ, fh)


            except Exception as e:
                print("无法运行面板 %d" % (panel_idx))
                print(e)
                exit()
                traceback.print_exc()

    def _get_cumulative_germ(self, germ, win=5):
        for m in range(germ.shape[0]):
            curr_seed = germ[m, :]
            idx = 0
            while idx < (curr_seed.shape[0] - win):
                if curr_seed[idx:idx + win].all():
                    curr_seed[:idx + win - 1] = 0
                    curr_seed[idx + win - 1:] = 1
                    break
                idx += 1
            if idx >= (curr_seed.shape[0] - win):
                curr_seed[:] = 0
        return germ.sum(axis=0), germ

    def _analyse_results(self, proprtions):
        all_germ = []
        for i in range(self.exp.panel_n):
            # 如果with不能打开，我们不应该对胚芽进行操作。
            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (i))) as fh:
                germ_d = json.load(fh)

                # 确保胚芽不是空的。
                if len(germ_d) == 0:
                    continue

                germinated = [germ_d[str(j)] for j in range(1, len(germ_d.keys()) + 1)]
                germinated = np.vstack(germinated)
                all_germ.append(germinated)

        p_totals = []
        for i in range(self.exp.panel_n):
            # l, rprop = self.panel_l_rprops[i]
            l, rprop = self.all_rprops[0][i]
            p_totals.append(len(rprop))

        if not all_germ:
            raise Exception("找到的发芽种子为0。尝试更改YUV值。")

        print(p_totals)

        np.save(self.exp_results_dir + '/all_germ.npy', all_germ)

        cum_germ_data = [
            self._get_cumulative_germ(germ, win=4)[0] for germ in all_germ
        ]

        initial_germ_time_data = []

        for germ in all_germ:
            rows, cols = germ.shape
            init_germ_time = []
            for m in range(rows):
                for n in range(cols):
                    if germ[m, n]:
                        # 用于添加偏移
                        init_germ_time.append(n + self.exp.start_img)
                        break  # 也许这就是问题的根源
            initial_germ_time_data.append(init_germ_time)

        for i in range(len(cum_germ_data)):
            cum_germ_data[i] = pd.Series(cum_germ_data[i])

        cum_germ_data = pd.concat(cum_germ_data, axis=1).astype('f')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                     figsize=(18., 15.),
                                                     dpi=650)

        fig.suptitle(self.exp.name)

        p_t_text = ""
        for i in range(self.exp.panel_n):
            p_t_text += "panel %d: %d" % (i + 1, p_totals[i])
            p_t_text += "\n" if ((i + 1) % 2) == 0 else "    "
        plt.figtext(0.05, 0.93, p_t_text)

        # 仅当文件名中包含日期信息时才使用
        has_date = check_files_have_date(self.imgs[0])
        start_dt = None
        if has_date:
            start_dt = s_to_datetime(self.imgs[0])

        n_frames = cum_germ_data.shape[0]

        for idx in range(cum_germ_data.shape[1]):
            ax1.plot(
                range(self.exp.start_img, n_frames + self.exp.start_img),
                cum_germ_data.iloc[:, idx] / float(p_totals[idx]), label="Genotype" + str(idx + 1)
            )
        ax1.set_xlim([self.exp.start_img, self.exp.start_img + n_frames])

        if has_date:
            # Sort out xtick labels in hours
            xtick_labels = []
            for val in ax1.get_xticks():
                if int(val) >= (self.exp.end_img - self.exp.start_img):
                    break
                curr_dt = s_to_datetime(self.imgs[int(val)])
                xtick_labels.append(hours_between(start_dt, curr_dt, round_minutes=True))

            ax1.set_xlabel("时间（小时）")
            ax1.set_xticklabels(xtick_labels, )
        else:
            ax1.set_xlabel("图像ID")

        ax1.legend(loc="upper left")
        ax1.set_ylabel("累计发芽率")
        ax1.set_title("累积发芽率")
        ax1.grid()

        data = []
        for idx in range(cum_germ_data.shape[1]):
            cum_germ = cum_germ_data.iloc[:, idx].copy().ravel()
            cum_germ /= p_totals[idx]

            germ_pro_total = cum_germ[-1]
            prop_idxs = []
            for pro in proprtions:
                if (cum_germ > pro).any():
                    pos_idx = np.argmax(cum_germ >= pro) + self.exp.start_img

                    if has_date:
                        curr_dt = s_to_datetime(
                            self.imgs[self.exp.start_img + pos_idx])
                        pos_idx = hours_between(start_dt, curr_dt)
                else:
                    pos_idx = 'n/a'
                prop_idxs.append(str(pos_idx))
            data.append(prop_idxs)

        columns = tuple('%d%%' % (100 * prop) for prop in proprtions)
        rows = ['  %d  ' % (x) for x in range(1, self.exp.panel_n + 1)]
        the_table = ax2.table(cellText=data,
                              rowLabels=rows,
                              colLabels=columns,
                              loc='center')

        tbl_props = the_table.properties()
        tbl_cells = tbl_props['children']
        for cell in tbl_cells:
            cell.set_height(0.1)
        ax2.set_title("T值百分比")
        ax2.axis('off')

        # Old code for setting an end point when roots overlap
        # for i in range(len(initial_germ_time_data)):
        #     for j in range(len(initial_germ_time_data[i]) - 1, -1, -1):
        #         if initial_germ_time_data[i][j] > self.end_idx[i]:
        #             initial_germ_time_data[i].pop(j)
        ax3.boxplot(initial_germ_time_data, vert=False
                    # , whis='range'
                    )
        ax3.set_xlim([self.exp.start_img, self.exp.start_img + n_frames + 5])
        ax3.set_ylabel("面板编号")
        ax3.set_title('发芽均匀度箱形图')
        ax3.grid()

        if has_date:
            ax3.set_xlabel("时间（小时）")
            xtick_labels = []
            for val in ax3.get_xticks():
                if int(val) >= len(self.imgs):
                    break
                curr_dt = s_to_datetime(self.imgs[int(val)])
                xtick_labels.append(hours_between(start_dt, curr_dt, round_minutes=True))
            ax3.set_xticklabels(xtick_labels)
        else:
            ax3.set_xlabel("图像ID")

        print(cum_germ_data.iloc[-1, :] / np.array(p_totals))

        # ax4.barh(np.arange(self.exp.panel_n) + 0.75, np.flipud((cum_germ_data.max(axis=0) / np.array(p_totals)).values.reshape(-1, 1)).ravel(), height=0.5)
        ax4.barh(np.arange(self.exp.panel_n) + 0.75, (cum_germ_data.max(axis=0) / np.array(p_totals)), height=0.5)
        ax4.set_yticks(range(1, 1 + self.exp.panel_n))
        ax4.set_ylim([0.5, self.exp.panel_n + .5])
        ax4.set_xlim([0., 1.])
        ax4.set_ylabel("面板编号")
        # ax4.set_xlabel("Germinated proportion")
        ax4.set_xlabel("发芽率")
        # ax4.set_title("Proportion germinated")
        ax4.set_title("发芽率")

        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax4.set_ylim(ax4.get_ylim()[::-1])
        fig.savefig(pj(self.exp_results_dir, "results.jpg"))
        plt.close(fig)

        img_index = np.arange(n_frames) + self.exp.start_img

        if has_date:
            times_index = []
            for _i in img_index:
                curr_dt = s_to_datetime(self.imgs[_i])
                times_index.append(hours_between(start_dt, curr_dt))

            times_index = np.array(times_index).reshape(-1, 1)
            cum_germ_data = np.hstack([times_index, cum_germ_data.values])

        df = pd.DataFrame(data=cum_germ_data)
        df.index = img_index

        if has_date:
            df.columns = ["Time"] + [str(i) for i in range(1, self.exp.panel_n + 1)]
            df.loc['Total seeds', 1:] = p_totals
        else:
            df.columns = [str(i) for i in range(1, self.exp.panel_n + 1)]
            df.loc['Total seeds', :] = p_totals

        df.to_csv(pj(
            self.exp_results_dir,
            "panel_germinated_cumulative.csv"
        ))

    def _quantify_first_frame(self, proprtions):
        """ 量化第一帧的种子数据。
        量化：
            - 种子总数
            - 种子分析
            - 初始种子大小
            - 初始种子圆度
            - 宽高比
            - RGB平均值
            - 不同百分比的发芽率
            - 种子x，y
        """

        # 仅当文件名中包含日期信息时才使用
        has_date = check_files_have_date(self.imgs[0])
        start_dt = None
        if has_date:
            start_dt = s_to_datetime(self.imgs[0])

        img_f = self.all_imgs_list[self.exp.start_img]
        f_masks = np.load(self.exp_masks_dir_frame % (self.exp.start_img), allow_pickle=True)

        img_l = self.all_imgs_list[self.exp.end_img - 1]
        l_masks = np.load(self.exp_masks_dir_frame % ((self.exp.end_img - self.exp.start_img) - 1), allow_pickle=True)

        all_panel_data = []

        # 面板分析
        for p_idx, (p_labels, p_rprops) in enumerate(self.panel_l_rprops):

            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (p_idx))) as fh:
                germ_d = json.load(fh)

            germinated = [germ_d[str(j)] for j in range(1, len(germ_d.keys()) + 1)]
            germinated = np.vstack(germinated)

            cum_germ = self._get_cumulative_germ(germinated, win=7)[0].astype('f')
            cum_germ /= len(p_rprops)

            germ_pro_total = cum_germ[-1]

            prop_idxs = []
            for pro in proprtions:
                if (cum_germ > pro).any():
                    pos_idx = np.argmax(cum_germ >= pro) + self.exp.start_img
                    if has_date:
                        curr_dt = s_to_datetime(self.imgs[pos_idx])
                        pos_idx = hours_between(start_dt, curr_dt)
                else:
                    pos_idx = 'n/a'
                prop_idxs.append(pos_idx)

            p_f_img = self.panel_list[p_idx].get_bbox_image(img_f)
            p_f_mask = f_masks[p_idx]

            p_l_img = self.panel_list[p_idx].get_bbox_image(img_l)
            p_l_mask = l_masks[p_idx]

            f_rgb_mu = p_f_img[p_f_mask].mean(axis=0)
            l_rgb_mu = p_l_img[p_l_mask].mean(axis=0)
            f_rgb_mu = tuple(np.round(f_rgb_mu).astype('i'))
            l_rgb_mu = tuple(np.round(l_rgb_mu).astype('i'))

            avg_feas = []
            for rp in p_rprops:
                min_row, min_col, max_row, max_col = rp.bbox
                w = float(max_col - min_col)
                h = float(max_row - min_row)
                whr = w / h
                avg_feas.append([w, h, whr, rp.area, rp.eccentricity])

            avg_feas = np.vstack(avg_feas)
            avg_feas_mu = avg_feas.mean(axis=0)

            panel_data = [p_idx + 1, len(p_rprops)]
            panel_data += np.round(avg_feas_mu, 2).tolist()
            panel_data += [f_rgb_mu, l_rgb_mu]
            panel_data += prop_idxs
            panel_data += [round(germ_pro_total, 2)]

            all_panel_data.append(panel_data)

        columns = [
            'panel_ID',
            'total_seeds',
            'avg_width',
            'avg_height',
            'avg_wh_ratio',
            'avg_area',
            'avg_eccentricity',
            'avg_initial_rgb',
            'avg_final_rgb',
            *['germ_%d%%' % (100 * prop) for prop in proprtions],
            'total_germ_%',
        ]

        df = pd.DataFrame(all_panel_data, columns=columns)
        df.to_csv(pj(self.exp_results_dir, "overall_results.csv"), index=False)

        # 种子分析
        all_seed_results = []

        panel_seed_idxs = {}

        for p_idx, (p_labels, p_rprops) in enumerate(self.panel_l_rprops):

            panel_seed_idxs[int(p_idx)] = []

            with open(pj(self.exp_results_dir, "germ_panel_%d.json" % (p_idx))) as fh:
                germ_d = json.load(fh)

            germinated = [germ_d[str(j)] for j in range(1, len(germ_d.keys()) + 1)]
            germinated = np.vstack(germinated)

            cum_germ, germ_proc = self._get_cumulative_germ(germinated, win=7)

            for seed_rp in p_rprops:

                germ_row = germ_proc[seed_rp.label - 1]

                germ_idx = 'n/a'
                if germ_row.any():
                    germ_idx = np.argmax(germ_row) + self.exp.start_img

                min_row, min_col, max_row, max_col = seed_rp.bbox
                w = float(max_col - min_col)
                h = float(max_row - min_row)
                whr = w / h

                if germ_idx == 'n/a':
                    germ_time = 'n/a'
                else:
                    if has_date:
                        curr_dt = s_to_datetime(self.imgs[germ_idx])
                        germ_time = hours_between(start_dt, curr_dt)
                    else:
                        germ_time = germ_idx

                seed_result = [
                    p_idx + 1,
                    seed_rp.label,
                    int(w),
                    int(h),
                    round(whr, 2),
                    int(seed_rp.area),
                    round(seed_rp.eccentricity, 2),
                    # (0,0,0),
                    # (0,0,0),
                    germ_idx,
                    germ_time,
                ]

                if germ_idx == 'n/a':
                    germ_idx = -1

                panel_seed_idxs[int(p_idx)].append((
                    int(germ_idx),
                    int(seed_rp.centroid[0]),
                    int(seed_rp.centroid[1])
                ))
                all_seed_results.append(seed_result)

        columns = [
            'panel_ID',
            'seed_ID',
            'width',
            'height',
            'wh_ratio',
            'area',
            'eccentricity',
            # 'initial_rgb',
            # 'final_rgb',
            'germ_point',
            'germ_time' if has_date else 'germ_image_number',
        ]

        df = pd.DataFrame(all_seed_results, columns=columns)
        df.to_csv(pj(self.exp_results_dir, "panel_results.csv"), index=False)

        with open(pj(self.exp_results_dir, "panel_seed_idxs.json"), "w") as fh:
            json.dump(panel_seed_idxs, fh)

    def run(self):
        print("开始分析")

        if self.running:

            start = time.time()

            try:
                self._save_init_image(self.imgs[self.exp.start_img])  # 保存初始图象

                self._extract_panels(self.imgs[self.exp.start_img], self.core.chunk_no, self.core.chunk_reverse,
                                     self.exp.start_img)  # 提取面板

                self.app.status_string.set("保存轮廓图像")

                self._save_contour_image()

                # self.app.status_string.set("Training background removal clfs")
                self.app.status_string.set("训练移除种子背景")

                # 用实验装置中定义的分类器进行种子分割
                if self.exp.bg_remover == 'UNet':
                    self.app.status_string.set("正在移除背景")
                    self._train_unet()
                elif self.exp.bg_remover == "SGD":
                    # 定义随机梯度下降分类器的超参数
                    self._train_clfs([("SGD", SGDClassifier(max_iter=50, random_state=0, tol=1e-5))])
                    self.app.status_string.set("正在移除背景")
                    self._remove_background()
                elif self.exp.bg_remover == "GMM":
                    self._train_gmm_clfs()
                    self.app.status_string.set("正在移除背景")
                    self._gmm_remove_background()
                else:
                    print("未定义的背景分类器")

                self.app.status_string.set("标记种子")
                self._label_seeds()

                self.app.status_string.set("生成统计信息")
                self._generate_statistics()

                self.app.status_string.set("正在执行分类")
                self._perform_classification()

                self.app.status_string.set("正在分析结果")

                self._analyse_results(self.core.proportions)

                self.app.status_string.set("量化初始种子数据")
                self._quantify_first_frame(self.core.proportions)

                self.app.status_string.set("已完成分析")
                self.exp.status = "完成"

                print("结束值：", self.end_idx + self.exp.start_img)

                print('分析用时：', time.time() - start)

                print("完成")

            except Exception as e:
                raise e
            self.running = False

            self.core.stop_processor(self.exp.eid)

    def die(self):
        self.running = False
