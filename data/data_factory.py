#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: data_factory.py
@time: 18-3-2 上午10:58
@desc:
'''

import numpy as np

def labels_img_to_colors(img):
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

    img_color = np.zeros((img_height, img_width))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]

            img_color[row, col] = np.array(label_colors[label])
    img_color = img_color[:, :, ::-1]

    return img_color
