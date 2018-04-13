#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: resnet_utils.py
@time: 18-2-27 下午2:53
@desc:
'''
import tensorflow as tf
from tensorflow.contrib import slim

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def subsample(inputs, factor, scope=None):
    """
    Subsamples the input along the spatial dimensions.
    :param inputs: A ‘Tensor’ of size [batch, height_in, width_in, channels].
    :param factor: The subsampling factor.
    :param scope: Optional variable_scope.
    :return:
        output: A ‘Tensor’ of size [batch_size, height_in, height_out, channels] with the input, either intact (if
        factor == 1) or subsampled (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D convolution with 'SAME' padding.
        When stride > 1, then we do explicit zero-padding, followed by conv2d with
        'VALID' padding.
        Note that
            net = conv2d_same(inputs, num_outputs, 3, stride=stride)
        is equivalent to
            net = slim.conv2d(inputs, num_outputs, 3, stride=1,
            padding='SAME')
            net = subsample(net, factor=stride)
        whereas
            net = slim.conv2d(inputs, num_outputs, 3, stride=stride,
            padding='SAME')
        is different when the input's height or width is even, which is why we add the
        current function. For more details, see ResnetUtilsTest.testConv2DSameEven().
        Args:
            inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
            num_outputs: An integer, the number of output filters.
            kernel_size: An int with the kernel_size of the filters.
            stride: An integer, the output stride.
            rate: An integer, rate for atrous convolution.
            scope: Scope.
        Returns:
            output: A 4-D tensor of size [batch, height_out, width_out, channels] with
            the convolution output.
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)

def build_block(inputs, base_depth, name, stride=2, rate=1):
    """
    residual block unit variant with BN after convolutions.
    :param inputs: A tensor of size [batch, height, width, channels].
    :param base_depth: the depth of the first unit of residual block where th output depth of this residual block is
    base_depth * 4.
    :param name: Optional variable_scope.
    :param stride: The ResNet unit's stride. Determines the amount of downsampling of the units output compared to its
    input.
    :param rate: An integer, rate for atrous convolution.
    :return:
        The ResNet unit's output.
    """
    depth_in = inputs.get_shape().as_list()[-1]
    # print("depth_in:", depth_in)
    if depth_in == base_depth * 4:
        shortcut = subsample(inputs, stride, 'shortcut')
    else:
        shortcut = slim.conv2d(inputs, num_outputs=base_depth * 4, kernel_size=1, stride=1, activation_fn=None,
                               scope='shortcut')
    # print("shortcut:", shortcut.shape)

    net = slim.conv2d(inputs, num_outputs=base_depth, kernel_size=1, stride=1, scope=name+"conv1")
    net = conv2d_same(net, num_outputs=base_depth, kernel_size=3, stride=stride, rate=rate, scope=name+"conv2")
    net = slim.conv2d(net, num_outputs=base_depth*4, kernel_size=1, stride=1, scope=name+"conv3")
    # print("net:", net.shape)

    # output = slim.relu(shortcut + net)
    return net + shortcut

def residual_layer(inputs, base_depth, num_units, rate=1, stride=1):
    """
    Create one layer of the normal residual blocks for the ResNet.
    :param inputs: A Tensor of size [batch, height_in, width_in, channels].
    :param base_depth: The input depth of the bottleneck layer for each unit.
    :param num_units: the number of blocks of this layer.
    :param rate: An integer, rate for atrous convolution.
    :param stride: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
    :return:
        the output tensor of the block layer.
    """
    for i in range(num_units):
        inputs = build_block(inputs, base_depth, name="unit_%d" % (i + 1), stride=stride, rate=rate)
    return inputs

def residual_atrous_layer(inputs, base_depth, num_units, stride, rate):
    """
    Create one layer pf the residual block with atrous convolution.
    :param inputs: A Tensor of size [batch, height_in, width_in, channels].
    :param base_depth: The input depth of the bottleneck layer for each unit.
    :param num_units: the number of blocks of this layer.
    :param rate: An integer, rate for atrous convolution.
    :param stride: The stride to use for the first convolution of the layer. If
        greater than 1, this layer will ultimately downsample the input.
    :return:
        the output tensor of the block layer.
    """
    for i in range(num_units):
        inputs = build_block(inputs, base_depth, name="unit_%d" % (i + 1), stride=stride, rate=rate[i])
    return inputs

def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs

def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
    """Defines the default ResNet arg scope.
    TODO(gpapan): The batch-normalization related default values above are
    appropriate for use in conjunction with the reference ResNet models
    released at https://github.com/KaimingHe/deep-residual-networks. When
    training ResNets from scratch, they might need to be tuned.
    Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    activation_fn: The activation function which is used in ResNet.
    use_batch_norm: Whether or not to use batch normalization.
    Returns:
    An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          # The following implies padding='SAME' for pool1, which makes feature
          # alignment easier for dense prediction tasks. This is also used in
          # https://github.com/facebook/fb.resnet.torch. However the accompanying
          # code of 'Deep Residual Learning for Image Recognition' uses
          # padding='VALID' for pool1. You can switch to that choice by setting
          # slim.arg_scope([slim.max_pool2d], padding='VALID').
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
