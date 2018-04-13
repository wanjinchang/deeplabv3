#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: data_preprocess.py
@time: 18-3-1 下午4:15
@desc:
'''

import cv2
import numpy as np
import tensorflow as tf

# color map for the segmentation image
label_colors = [
                 [0, 0, 0],        # 0=background
                 [128, 0, 0],      # 1=aeroplane
                 [0, 128, 0],      # 2=bicycle
                 [128, 128, 0],    # 3=bird
                 [0, 0, 128],      # 4=boat
                 [128, 0, 128],    # 5=bottle
                 [0, 128, 128],    # 6=bus
                 [128, 128, 128],  # 7=car
                 [64, 0, 0],       # 8=cat
                 [192, 0, 0],      # 9=chair
                 [64, 128, 0],     # 10=cow
                 [192, 128, 0],    # 11=diningtable
                 [64, 0, 128],     # 12=dog
                 [192, 0, 128],    # 13=horse
                 [64, 128, 128],   # 14=motorbike
                 [192, 128, 128],  # 15=person
                 [0, 64, 0],       # 16=potted plant
                 [128, 64, 0],     # 17=sheep
                 [0, 192, 0],      # 18=sofa
                 [128, 192, 0],    # 19=train
                 [0, 64, 128]      # 20=tv/monitor
                 ]
def color_map():
    return tf.cast(tf.stack(label_colors), tf.float32)

def color_mask(tensor, color):
    """
    compare the input tensor with the color map.
    :param tensor: a Tensor with [batch, image_height, image_width, channels]
    :param color: a Tensor with [channels]
    :return:
        tf.bool to indicate the comparsion of the input tensor and the color map.
    """
    return tf.reduce_all(tf.equal(tensor, color), axis=3)

def one_hot(labels):
    """
    get the one-hot encoding for the ground truth labels.
    :param labels: grouth truth labels. [batch, image_height, image_width, channels]
    :return: one-hot encoding labels.
    """
    colors = color_map()
    color_tensors = tf.unstack(colors)
    channel_tensors = list(map(lambda color: color_mask(labels, color), color_tensors))
    one_hot_labels = tf.cast(tf.stack(channel_tensors, 3), 'float32')
    return one_hot_labels

def horizontal_flip(image, gt_mask, axis):
    """
    Flip an image at 50% possibility.
    :param image: a 3 dimension numpy array representing an image.
    :param axis: 0 for vertical flip an 1 for horizontal flip.
    :return: 3D image after flip.
    """
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)
        gt_mask = cv2.flip(gt_mask, axis)

    return image, gt_mask

def rescale(image, gt_mask, scale):
    """
    scale the image and the gt_mask for data augmentation.
    :param image: a 3 dimension numpy array representing an image.
    :param gt_mask: a 3 dimension numpy array representing an gt_mask.
    :param scale: the scale factor.
    :return:
        image: numpy array of size of [height, width, 3]
        gt_mask: numpy array of size of [height, width, 3]
    """
    h = image.shape[0]
    w = image.shape[1]
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)
    # print("gt_mask:", gt_mask.shape[0], gt_mask.shape[1])
    new_image = cv2.resize(image, (scaled_h, scaled_w), interpolation=cv2.INTER_LINEAR)
    new_mask = cv2.resize(gt_mask, (scaled_h, scaled_w), interpolation=cv2.INTER_NEAREST)
    return new_image, new_mask


def labels_img_to_colors(img, num_classes):
    """
    color map to assign to the labeled image.
    """
    label_colors = {
        0: (0, 0, 0),         # 0=background
        1: (128, 0, 0),       # 1=aeroplane
        2: (0, 128, 0),       # 2=bicycle
        3: (128, 128, 0),     # 3=bird
        4: (0, 0, 128),       # 4=boat
        5: (128, 0, 128),     # 5=bottle
        6: (0, 128, 128),     # 6=bus
        7: (128, 128, 128),   # 7=car
        8: (64, 0, 0),        # 8=cat
        9: (192, 0, 0),       # 9=chair
        10: (64, 128, 0),     # 10=cow
        11: (192, 128, 0),    # 11=diningtable
        12: (64, 0, 128),     # 12=dog
        13: (192, 0, 128),    # 13=horse
        14: (64, 128, 128),   # 14=motorbike
        15: (192, 128, 128),  # 15=person
        16: (0, 64, 0),       # 16=potted plant
        17: (128, 64, 0),     # 17=sheep
        18: (0, 192, 0),      # 18=sofa
        19: (128, 192, 0),    # 19=train
        20: (0, 64, 128)      # 20=tv/monitor
    }
    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            if label <= (num_classes - 1):
                img_color[row, col] = np.array(label_colors[label])

    return img_color

if __name__ == '__main__':
    file_name = "/media/thinkjoy/0000678400004823/Segment_Data/VOCdevkit/VOC2012/SegmentationClass/2011_002956.png"
    img = cv2.imread(file_name)
    img = np.expand_dims(img, axis=0)
    inputs = tf.placeholder(tf.float32, [None, 512, 512, 3])
    with tf.Session() as sess:
        sess.run(one_hot, feed_dict={inputs: img})



