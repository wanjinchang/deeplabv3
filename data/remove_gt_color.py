#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: remove_gt_color.py
@time: 18-4-9 下午3:20
@desc:
'''
import glob
import os
import numpy as np

from PIL import Image

import tensorflow as tf

def _remove_colormap(filename):
    """
    Removes the color map from the annotation.
    :param filename: Ground truth annotation filename.
    :return:
        Annotation without color map.
    """
    return np.array(Image.open(filename))

def _save_annotation(annotation, filename):
    """
    Saves the annotation as png file.
    :param annotation: Segmentation annotation.
    :param filename: Output filename.
    """
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.gfile.Open(filename, mode='w') as f:
        pil_image.save(f, 'png')

def main(output_dir, original_dir):
    # Create the output directory if not exists.
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)

    annotations = os.listdir(original_dir)

    for annotation in annotations:
        # print("processing on: ", annotation)
        raw_annotation = _remove_colormap(os.path.join(original_dir, annotation))
        filename = os.path.basename(annotation)[:-4]
        _save_annotation(raw_annotation, os.path.join(output_dir, filename + '.' + 'png'))

if __name__ == '__main__':
    output_dir = "/media/thinkjoy/0000678400004823/Segment_Data/VOCdevkit/VOC2012/SegmentationRaw/"
    original_dir = "/media/thinkjoy/0000678400004823/Segment_Data/VOCdevkit/VOC2012/SegmentationClass/"
    main(output_dir, original_dir)
