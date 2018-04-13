#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: deeplab_resnet_101.py
@time: 18-2-27 上午11:58
@desc:
'''

from resnet_utils import *
import tensorflow as tf
from tensorflow.contrib import slim


class DeepLabv3(object):

    def __init__(self, inputs, num_classes=21, weight_decay=2e-4, aspp=True, is_training=False):
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.weight_decay = weight_decay
        self.blocks = {"block1": {"base_depth": 64, "num_units": 3, "strides": 2},
                       "block2": {"base_depth": 128, "num_units": 4, "strides": 2},
                       "block3": {"base_depth": 256, "num_units": 23, "strides": 2},
                       "block4": {"base_depth": 512, "num_units": 3, "strides": 1},
                       "block5": {"base_depth": 512, "num_units": 3, "strides": 1},
                       "block6": {"base_depth": 512, "num_units": 3, "strides": 1},
                       "block7": {"base_depth": 512, "num_units": 3, "strides": 1},
                       }
        self.aspp = aspp
        self.model_name = "deeplab_resnet101"

    def ASPP(self, inputs, dilated_series, output_depth):
        """
        Implementation of the Atrous Spatial Pyramid Pooling described of DeepLabv3.
        :param inputs: A Tensor of size [batch, height_in, width_in, channels].
        :param dilated_series: A tuple of the atrous rate.
        :param output_depth: The output depth of the layer.
        :return:
            aspp_list: A list contain the feature map Tensor after aspp.
        """
        with tf.variable_scope("aspp"):
            aspp_list = []
            branch_1 = slim.conv2d(inputs, num_outputs=output_depth, kernel_size=1, stride=1, scope="conv_1x1")
            aspp_list.append(branch_1)

            for i in range(3):
                branch_2 = slim.conv2d(inputs, num_outputs=output_depth, kernel_size=3, stride=1, rate=dilated_series[i],
                                       scope="rate{}".format(dilated_series[i]))
                aspp_list.append(branch_2)

            return aspp_list


    def build_model(self):
        end_points = {}
        if self.aspp:
            multi_grid = (1, 2, 4)
        else:
            multi_grid = (1, 2, 1)
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=self.is_training):
            with tf.variable_scope(self.model_name):
                with slim.arg_scope(resnet_arg_scope(weight_decay=self.weight_decay)):
                    with tf.variable_scope("init_block"):
                        net = conv2d_same(self.inputs, num_outputs=64, kernel_size=7, stride=2, scope="conv1")
                        # print("net:", net.shape)
                        net = slim.max_pool2d(net, kernel_size=3, stride=2, scope='pool1')
                        # print("init_block:", net.shape)
                        self.init_block = net
                        end_points["init_block"] = self.init_block

                    with tf.variable_scope("block1"):
                        net = residual_layer(net, base_depth=self.blocks["block1"]["base_depth"],
                                          num_units=self.blocks["block1"]["num_units"] - 1, stride=1, rate=1)

                        # the stride of the last unit of block1 should be 2
                        net = build_block(net, base_depth=self.blocks["block1"]["base_depth"],
                                          name="units_%d" % self.blocks["block1"]["num_units"], stride=2, rate=1)
                        block1 = net
                        end_points["block1"] = block1

                    with tf.variable_scope("block2"):
                        net = residual_layer(net, base_depth=self.blocks["block2"]["base_depth"],
                                          num_units=self.blocks["block2"]["num_units"], stride=1, rate=1)

                        # the stride of the last unit of block2 should be 2
                        net = build_block(net, base_depth=self.blocks["block2"]["base_depth"],
                                          name="units_%d" % self.blocks["block2"]["num_units"], stride=2, rate=1)
                        block2 = net
                        end_points["block2"] = block2

                    with tf.variable_scope("block3"):
                        # specify the rate=2, use the atrous convolution
                        net = residual_layer(net, base_depth=self.blocks["block3"]["base_depth"],
                                          num_units=self.blocks["block3"]["num_units"], stride=1, rate=1)
                        block3 = net
                        end_points["block3"] = block3

                    with tf.variable_scope("block4"):
                        # specify the rate=2, use the atrous convolution
                        net = residual_atrous_layer(net, base_depth=self.blocks["block4"]["base_depth"],
                                          num_units=self.blocks["block4"]["num_units"], stride=1, rate=2*multi_grid)
                        block4 = net
                        end_points["block4"] = block4

                    # if self.aspp=True, then make Atrous Spatial Pyramid Pooling module, otherwise make block5~block7
                    if self.aspp:
                        aspp_list = self.ASPP(net, [6, 12, 18], 256)
                        end_points["aspp1"] = aspp_list[0]
                        end_points["aspp2"] = aspp_list[1]
                        end_points["aspp3"] = aspp_list[2]
                        end_points["aspp4"] = aspp_list[3]

                        # Image Pooling
                        with tf.variable_scope("img_pool"):
                            # print("net:", net.shape)
                            pooled = tf.reduce_mean(net, [1, 2], name="avg_pool", keep_dims=True)
                            # end_points["aspp5"] = pooled

                            global_feat = slim.conv2d(pooled, num_outputs=256, kernel_size=1, stride=1, scope="conv1x1")
                            global_feat = tf.image.resize_bilinear(global_feat, tf.shape(net)[1:3])
                            # print("global_feat:", global_feat.shape)
                            aspp_list.append(global_feat)
                            end_points["aspp5"] = global_feat

                        # Concat the output of the aspp module
                        net = tf.concat(aspp_list, axis=3)
                        end_points['fusion'] = net

                        # with tf.variable_scope("fusion"):
                        #     net = tf.concat(aspp_list, 3)
                        #
                        #     net = slim.conv2d(net, num_outputs=256, kernel_size=1, stride=1, scope="conv1x1")
                        #     end_points["fusion"] = net
                    else:
                        with tf.variable_scope("block5"):
                            net = residual_atrous_layer(net, base_depth=self.blocks["block5"]["base_depth"],
                                              num_units=self.blocks["block5"]["num_units"], stride=1, rate=4*multi_grid)
                            block5 = net
                            end_points["block5"] = block5

                        with tf.variable_scope("block6"):
                            net = residual_atrous_layer(net, base_depth=self.blocks["block6"]["base_depth"],
                                              num_units=self.blocks["block6"]["num_units"], stride=1,
                                              rate=4 * multi_grid)
                            block6 = net
                            end_points["block6"] = block6

                        with tf.variable_scope("block7"):
                            net = residual_atrous_layer(net, base_depth=self.blocks["block7"]["base_depth"],
                                              num_units=self.blocks["block7"]["num_units"], stride=1,
                                              rate=4 * multi_grid)
                            block7 = net
                            end_points["block7"] = block7

                    with tf.variable_scope("logits"):
                        net = slim.conv2d(net, num_outputs=self.num_classes, kernel_size=1, stride=1,
                                          activation_fn=None, normalizer_fn=None)
                        logits = net
                        end_points["logits"] = logits
                    return net, end_points

class DeepLabv4(object):
    def __init__(self, inputs, num_classes=21, weight_decay=2e-4, aspp=True, is_training=False):
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.weight_decay = weight_decay
        self.blocks = {"block1": {"base_depth": 64, "num_units": 3, "strides": 2},
                       "block2": {"base_depth": 128, "num_units": 4, "strides": 2},
                       "block3": {"base_depth": 256, "num_units": 23, "strides": 2},
                       "block4": {"base_depth": 512, "num_units": 3, "strides": 1},
                       "block5": {"base_depth": 512, "num_units": 3, "strides": 1},
                       "block6": {"base_depth": 512, "num_units": 3, "strides": 1},
                       "block7": {"base_depth": 512, "num_units": 3, "strides": 1},
                       }
        self.aspp = aspp
        self.model_name = "deeplab_resnet101"

    def ASPP(self, inputs, dilated_series, output_depth):
        """
        Implementation of the Atrous Spatial Pyramid Pooling described of DeepLabv3.
        :param inputs: A Tensor of size [batch, height_in, width_in, channels].
        :param dilated_series: A tuple of the atrous rate.
        :param output_depth: The output depth of the layer.
        :return:
            aspp_list: A list contain the feature map Tensor after aspp.
        """
        with tf.variable_scope("aspp"):
            aspp_list = []
            branch_1 = slim.conv2d(inputs, num_outputs=output_depth, kernel_size=1, stride=1, scope="conv_1x1")
            aspp_list.append(branch_1)

            for i in range(3):
                branch_2 = slim.conv2d(inputs, num_outputs=output_depth, kernel_size=3, stride=1,
                                       rate=dilated_series[i],
                                       scope="rate{}".format(dilated_series[i]))
                aspp_list.append(branch_2)

        return aspp_list

    def build_model(self):
        """

        :return:
        """
        # TODO:not implemented...

if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, [None, 512, 512, 3])
    deeplabv3 = DeepLabv3(inputs)
    net, endpoints = deeplabv3.build_model()
    print('net:', net.shape)
    for i in endpoints.keys():
        print(i, endpoints[i].shape)



























