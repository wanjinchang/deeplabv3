#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: data_loader.py
@time: 18-3-6 上午11:55
@desc:
'''

import os
from data.data_preprocess import *
import cv2
import numpy as np

def make_dataset(root_dir, txt_path):
    path_list = []
    f = open(txt_path, "r")
    lines = f.readlines()
    for line in lines:
        data_path, gtmask_path = line.split(" ")
        gtmask_path = gtmask_path[:-1]
        # data_dir = os.path.join(root_dir, data_path)
        # gtmask_dir = os.path.join(root_dir, gtmask_path)
        data_dir = root_dir + data_path
        gtmask_dir = root_dir + gtmask_path
        item = (data_dir, gtmask_dir)
        path_list.append(item)
    np.random.shuffle(path_list)
    return path_list

class data_loader():
    def __init__(self, data_dir, txt_dir, img_height, img_width, batch_size):
        self.data_dir = data_dir
        self.txt_dir = txt_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.imgs = make_dataset(self.data_dir, self.txt_dir)

    def length(self):
        return len(self.imgs)

    def reader(self, idx):
        """
        Get the path of input image and ground truth mask.
        """
        # Get the path of input image and groundtruth mask.
        input_path, gtmask_path = self.imgs[idx]
        input_img, gt_img = self.loader(input_path, gtmask_path)
        return input_img, gt_img

    def loader(self, input_path, mask_path):
        """
        read the input image and the gt_mask as np.array.
        :param input_path: the path of the input image;
        :param mask_path: the path of the gt_mask image;
        :return:
            input_image: np.array;
            gt_mask: np.array.
        """
        input_image = cv2.imread(input_path)
        # h, w = input_image.shape
        # print("input_image:", h, w)
        # gt_mask = cv2.imread(mask_path)
        # bgr --> rgb
        # # input_image = input_image[:, :, ::-1]
        # gt_mask = gt_mask[:, :, ::-1]

        # the gt_mask should be gray image
        gt_mask = cv2.imread(mask_path, 0)
        # h, w = gt_mask.shape
        # print("gt_mask:", h, w)

        # randomly horizontal flip
        input_image, gt_mask = horizontal_flip(input_image, gt_mask, axis=1)

        # randomly scale
        scale = np.random.uniform(low=0.5, high=2.0, size=1)
        input_image, gt_mask = rescale(input_image, gt_mask, scale)

        input_image = cv2.resize(input_image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        gt_mask = cv2.resize(gt_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        # print('input_image:', input_image.shape)    # -> (512, 512, 3)
        # print('gt_mask:', gt_mask.shape)            # -> (512, 512, 3)
        gt_mask = np.expand_dims(gt_mask, axis=-1)
        return input_image, gt_mask

    def next_batch(self):
        """
        Next batch generator.
        Yields:
            input_images: Batch numpy array representation of batch of images;
            gt_masks: Batch numpy array representation of batch of gt_masks.
        """
        start_idx = 0
        while True:
            image_files = self.imgs[start_idx:start_idx + self.batch_size]
            # print('image_files:', image_files)

            # Read input_images and gt_masks from image_files
            images_batch = []
            gtmasks_batch = []
            for image_file in image_files:
                input_path, mask_path = image_file
                input_image, gt_mask = self.loader(input_path, mask_path)
                input_image = input_image.astype(np.float32) / 255.0
                gt_mask = gt_mask.astype(np.float32) / 255.0
                # np.asarray(input_image)
                # np.asarray(gt_mask)
                images_batch.append(input_image)
                gtmasks_batch.append(gt_mask)

            # images_batch = np.asarray(images_batch)
            # gtmasks_batch = np.asarray(gtmasks_batch)
            images_batch = np.array(images_batch)
            gtmasks_batch = np.array(gtmasks_batch)

            yield (images_batch, gtmasks_batch)

            # Update start index for the next batch
            start_idx += self.batch_size
            if start_idx >= len(self.imgs):
                start_idx = 0
