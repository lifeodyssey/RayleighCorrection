import logging
import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageFilter

import conf


def gen_gif(image_list, save_name):
    images = []
    for image_path in image_list:
        images.append(Image.open(image_path))

    images[0].save(save_name, save_all=True, append_images=images,
                   loop=0, duration=1000)

    return save_name


def gen_mp4(image_list, save_name):
    image0 = Image.open(image_list[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(save_name, fourcc, 1, image0.size)
    for image_path in image_list:
        frame = cv2.imread(image_path)
        vw.write(frame)
    vw.release()

    return save_name


def merge_pic(merge_save: str, bloom_index: str, blue_algae: str, **kwargs):
    """
    合并图片

    如果 bloom_index 和 blue_algae 都有对应文件存在，合并两个图片并返回；
    若只有 bloom_index 文件存在，则返回 bloom_index 文件路径；
    若只有 blue_algae 文件存在，则返回 blue_algae 文件路径.

    :param merge_save: 合并图片保存路径
    :param bloom_index: 水华指数图片路径
    :param blue_algae: 蓝藻范围图片路径
    :returns: 合并图片路径
    """
    if bloom_index is None and blue_algae is None:
        return None
    if bloom_index is None and blue_algae is not None:
        return blue_algae
    if blue_algae is None and bloom_index is not None:
        return bloom_index
    try:
        bloom_index_img = Image.open(bloom_index)
        blue_algae_img = Image.open(blue_algae)
        result = Image.blend(bloom_index_img, blue_algae_img, alpha=1)
        result.save(merge_save)
    except Exception as e:
        logging.error('image_util-merge_pic merge {} and {} has error, {}'.format(bloom_index, blue_algae, e))
        return None
    return merge_save


# 锐化
def sharpen(file_name, save_name):
    im = Image.open(file_name)
    sharpened = im.filter(ImageFilter.SHARPEN)
    sharpened2 = sharpened.filter(ImageFilter.SHARPEN)

    sharpened2.save(save_name)

    return save_name


# 伪色彩
def pseudo_color(file_name, save_name):
    im_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    cv2.imwrite(save_name, im_color)

    return save_name


# 加权平滑
def gray_weightmean(file_name, save_name):
    wr = 0.299
    wg = 0.587
    wb = 0.114
    img = cv2.imread(file_name)
    gray_wightmean_rgb_image = img.copy()
    img_shape = img.shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            gray_wightmean_rgb_image[i, j] = (int(wr * img[i, j][2]) + int(wg * img[i, j][1]) + int(
                wb * img[i, j][0])) / 3

    cv2.imwrite(save_name, gray_wightmean_rgb_image)
    return save_name


# 均值平滑
def blur(file_name, save_name):
    img = cv2.imread(file_name)
    img_blur = cv2.blur(img, (3, 3))
    # img_blur = cv2.blur(img, (5, 5))
    # img_blur = cv2.blur(img, (11, 11))

    cv2.imwrite(save_name, img_blur)
    return save_name


# 中值平滑
def medianBlur(file_name, save_name):
    img = cv2.imread(file_name)
    img_ret = cv2.medianBlur(img, 3)

    cv2.imwrite(save_name, img_ret)
    return save_name


# 黑白线性调色
def white_black(file_name, save_name):
    img = cv2.imread(file_name)
    cv2.imwrite(save_name, ~img)
    return save_name


# 曲线变换
def circle_fitness(file_name, save_name):
    img = cv2.imread(file_name, 0)
    # 灰度图专属
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w)).astype(np.uint8)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = 0.8 * (math.log(1.0 + img[i, j]))

    new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(save_name, new_img)
    return save_name


# 线性拉伸
def grey_scale(file_name, save_name):
    img = cv2.imread(file_name)
    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(img, table)

    cv2.imwrite(save_name, img)
    return save_name


#  均衡化
def equalizeHist(file_name, save_name):
    img = cv2.imread(file_name, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    cv2.imwrite(save_name, dst)
    return save_name


# 拉普拉斯增强
def filter(file_name, save_name):
    ori_gray = cv2.imread(file_name, 0)  # 读取灰度图像
    # ori = np.array(ori)
    ori_gray = np.array(ori_gray)
    weight = ori_gray.shape[0]
    height = ori_gray.shape[1]
    # laplation kernel
    h = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]), dtype="float32")
    filteredImg = cv2.filter2D(ori_gray, -1, h)
    cv2.imwrite(save_name, filteredImg)

    return save_name


def test_gen_mp4():
    base_path = r'/Users/yangwai/fsdownload/caijian_fire/high/source'
    file_list = os.listdir(base_path)
    _image_list = []
    save_path = '{}/{}_test.mp4'.format(conf.APP_DIR, time.time())
    for file in file_list:
        if file.rfind('.jpg') != -1 or file.rfind('.png') != -1:
            _image_list.append('{}/{}'.format(base_path, file))

    if len(_image_list) <= 0:
        return
    image_list = sorted(_image_list)

    if image_list is not None:
        gen_mp4(image_list=image_list, save_name=save_path)
        gen_gif(image_list=image_list, save_name=save_path.replace('mp4', 'gif'))


if __name__ == '__main__':
    test_gen_mp4()
    # pseudo_color('/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer1.png') gray_weightmean(
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer2.png') blur(
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer3.png') medianBlur(
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B04_Mer4.png') white_black(
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B04_Mer5.png')
    # circle_fitness('/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    #                '/Users/khahux/Desktop/tmp/China_202010150400_B04_Mer6.png')
    # grey_scale('/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B04_Mer7.png') equalizeHist(
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B04_Mer8.png') filter(
    # '/Users/khahux/Desktop/tmp/China_202010150400_B03_Mer.png',
    # '/Users/khahux/Desktop/tmp/China_202010150400_B04_Mer10.png')
