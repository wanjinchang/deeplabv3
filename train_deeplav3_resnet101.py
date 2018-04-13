#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: train_deeplav3_resnet101.py
@time: 18-3-8 下午3:24
@desc:
'''
from deeplab_resnet_101 import *
from data.data_loader import *
from tf_utils import *
import tensorflow as tf
import time
import datetime
from datetime import timedelta

# =================================================================================================================== #
# General Flags.
# =================================================================================================================== #
tf.app.flags.DEFINE_string('save_path', './logs/model/', 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('logdir_train', './logs/train', 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('logdir_test', './logs/test', 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

# =========================================================================== #
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')
tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5, 'The learning rate power.')
tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'Specifies how the learning rate is decayed. '
                                                                      'One of "fixed", "exponential",'
                                                                      ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 2.5e-4, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 5e-6, 'The minimal end learning rate used by a polynomial decay '
                                                       'learning rate.')
tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average.' 'If left as None, '
                                                        'then moving averages are not used.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer('num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string('dataset_dir', '/home/thinkjoy/sdb/VOC_Segment',
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('pred_dir', './pred_imgs',
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('train_txt', './train.txt', 'The txt file of train data.')
tf.app.flags.DEFINE_string('val_txt', './val.txt', 'The txt file of validation data.')
tf.app.flags.DEFINE_string(
    'model_name', 'DeepLabv3', 'The name of the architecture to train, should be one of ‘DeepLabv3’ or ‘DeepLabv3+’.')
tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'epochs', 200, 'The number of epoch for training.')
tf.app.flags.DEFINE_integer('img_height', 513, 'the height of the input image.')
tf.app.flags.DEFINE_integer('img_width', 513, 'the width of the input image.')

FLAGS = tf.app.flags.FLAGS

class Train_DeepLab():
    def __init__(self):
        self.flags = FLAGS
        self.img_height = self.flags.img_height
        self.img_width = self.flags.img_width
        self.num_classes = self.flags.num_classes
        self.batch_size = self.flags.batch_size
        self.nEpochs = self.flags.epochs
        # self.is_training = self.flags.is_training
        self.logdir_train = self.flags.logdir_train
        self.logdir_test = self.flags.logdir_test
        self.pred_dir = self.flags.pred_dir
        self.save_path = self.flags.save_path
        self.model_name = self.flags.model_name
        self.train_data_loader = data_loader(self.flags.dataset_dir, self.flags.train_txt,
                                             self.img_height, self.img_width, self.batch_size)
        self.num_samples_per_epoch = self.train_data_loader.length()
        self.val_data_loader = data_loader(self.flags.dataset_dir, self.flags.val_txt,
                                             self.img_height, self.img_width, self.batch_size)
        self._build_graph()
        self._initialize_session()
        self._create_dirs()

    def _initialize_session(self):
        """
        Initialize session, variables, saver
        """
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        # logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
        self.summary_writer_train = tf.summary.FileWriter(self.logdir_train, self.sess.graph)
        self.summary_writer_valid = tf.summary.FileWriter(self.logdir_test)

    def _create_dirs(self):
        self.model_dir = self.save_path
        self.checkpoints_dir = self.model_dir + "checkpoints"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.checkpoints_dir, global_step=global_step)

    def load_model(self):
        """
        load the sess from the pretrained model
        :return: start_epoch: the start step to train the model
        """
        ckpt = tf.train.get_checkpoint_state(self.save_path[0])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            start_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            start_epoch = int(start_epoch) + 1
            print('Successfully load model from save_path: %s and epoch: %s' % (self.save_path[0], start_epoch))
            return start_epoch
        else:
            print('Training from scratch')
            return 1

    def _build_graph(self):
        print('building graph!!')
        with tf.name_scope('inputs'):
            self.input_img = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_height, self.img_width, 3),
                                            name='input_image')
            self.gt_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_height, self.img_width, 1))
        if self.flags.model_name == 'DeepLabv3':
            self.DeepLab = DeepLabv3(self.input_img, self.num_classes, is_training=True)
        elif self.flags.model_name == 'DeepLabv4':
            self.DeepLab = DeepLabv4(self.input_img, self.num_classes, is_training=True)
        # Create network and Predictions
        net, _ = self.DeepLab.build_model()

        # Predictions: ignoring all predictions with labels greater or equal the num_classes
        # gt_mask = tf.squeeze(self.gt_mask)
        # print('gt_mask:', gt_mask.shape)
        # mask = gt_mask <= self.num_classes
        # gt_mask = tf.boolean_mask(gt_mask, mask)
        # gt_mask = tf.cast(gt_mask, tf.int32)
        # print('gt_mask:', gt_mask.shape)

        # gt_labels = one_hot(self.gt_mask)
        # mask = self.gt_mask <= FLAGS.num_classes

        # gt_labels = tf.reshape(gt_labels, [-1, self.num_classes])
        # print('gt_labels:', gt_labels.shape)

        # Unsample the logits instead of downsample the ground truth
        raw_output_up = tf.image.resize_bilinear(net, [self.img_height, self.img_width])
        # print('raw_output_up:', raw_output_up.shape)
        self.raw_logits = raw_output_up

        label_proc = tf.squeeze(self.gt_mask)
        mask = label_proc <= self.num_classes
        self.logits = tf.boolean_mask(raw_output_up, mask)
        # print('logits:', self.logits.shape)
        seg_gt = tf.boolean_mask(label_proc, mask)
        self.seg_gt = tf.cast(seg_gt, tf.int32)

        # raw_output_up = tf.image.resize_bilinear(net, [self.img_height, self.img_width])
        # self.logits = raw_output_up
        # print('raw_output_up:', raw_output_up.shape)
        # raw_output_up = tf.boolean_mask(raw_output_up, mask)
        # print('raw_output_up:', raw_output_up.shape)
        # logits = raw_output_up

        # Pixel-wise softmax loss.
        with tf.name_scope('seg_loss'):
            # loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_labels, logits=self.logits)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.seg_gt)
            self.seg_loss = tf.reduce_mean(loss)
        with tf.name_scope('reg_loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_losses = tf.add_n(reg_loss)
        # total loss
        with tf.name_scope('total_loss'):
            self.total_loss = self.seg_loss + self.reg_losses
        with tf.name_scope('steps'):
            self.train_step = tf.Variable(0, trainable=False, name='global_step')

        self.learning_rate = configure_learning_rate(self.flags, self.num_samples_per_epoch, self.train_step)
        self.optimizer = configure_optimizer(self.flags, self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.train_step)

        # Summary
        with tf.name_scope('train'):
            tf.summary.scalar('seg_loss', self.seg_loss, collections=['train'])
            tf.summary.scalar('reg_loss', self.reg_losses, collections=['train'])
            tf.summary.scalar('total_loss', self.total_loss, collections=['train'])
        with tf.name_scope('valid'):
            tf.summary.scalar('valid_loss', self.seg_loss, collections=['valid'])
        self.train_sum_op = tf.summary.merge_all('train')
        self.test_sum_op = tf.summary.merge_all('valid')

    def train_all_epochs(self):
        total_start_time = time.time()

        # Restore the model if we have
        start_epoch = self.load_model()

        # Start training
        for epoch in range(start_epoch, self.nEpochs + 1):
            print('Epoch :' + str(epoch) + '/' + str(self.nEpochs) + '\n')
            start_time = time.time()

            print('Training...')
            train_gen = self.train_data_loader.next_batch()
            val_gen = self.val_data_loader.next_batch()
            loss = self.train_one_epoch(epoch, train_gen)
            print('epoch: %d, train loss: %g' % (epoch, loss))
            time_per_epoch = time.time() - start_time
            second_left = int((self.nEpochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (str(timedelta(seconds=time_per_epoch)),
                                                                str(timedelta(seconds=second_left))))

            # Validation
            val_loss = self.eva_val(epoch, val_gen)
            print('Epoch: %d, mean_val_loss: %g' % (epoch, val_loss))
            if epoch % 10 == 0:
                self.save_model(global_step=epoch)

        total_train_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(seconds=total_train_time)))

    def train_one_epoch(self, epoch, gen):
        total_loss = []
        for step in range(self.train_data_loader.length() // self.batch_size):
            images_batch, gtmask_batch = next(gen)
            feed_dict = {self.input_img: images_batch, self.gt_mask: gtmask_batch}
            fetches = [self.train_op, self.total_loss, self.train_sum_op, self.train_step]
            _, loss, summary, n_step = self.sess.run(fetches, feed_dict=feed_dict)
            print('%s: Epoch: %d, Step: %d, loss: %6f' % (datetime.datetime.now(), epoch, step + 1, loss))
            total_loss.append(loss)
            self.summary_writer_train.add_summary(summary, n_step)
            # self.summary_writer_train.flush()
        mean_loss = np.mean(total_loss)
        return mean_loss

    def eva_val(self, epoch, gen):
        val_loss = []
        for step in range(self.val_data_loader.length() // self.batch_size):
            images_batch, gtmask_batch = next(gen)
            feed_dict = {self.input_img: images_batch, self.gt_mask: gtmask_batch}
            fetches = [self.total_loss, self.test_sum_op, self.raw_logits]
            loss, summary, logits = self.sess.run(fetches, feed_dict=feed_dict)
            if step % 20 == 0:
                print('Epoch: %d, val_step: %d, mean_val_loss: %6f' % (epoch, step + 1, loss))
            val_loss.append(loss)
            self.summary_writer_valid.add_summary(summary, epoch)
            # self.summary_writer_valid.flush()
            if step < 4:
                # save the prediction label img to disk
                predictions = np.argmax(logits, axis=3)
                for i in range(self.batch_size):
                    pred_img = predictions[i]
                    labled_img = labels_img_to_colors(pred_img, self.num_classes)
                    if not os.path.exists(self.pred_dir):
                        os.makedirs(self.pred_dir)
                    cv2.imwrite(self.pred_dir + "/" + "val_" + str(epoch) + "_" +
                                 str(step) + "_" + str(i) + ".png", labled_img)
        mean_val_loss = np.mean(val_loss)
        return mean_val_loss

if __name__ == '__main__':
    train_object = Train_DeepLab()
    train_object.train_all_epochs()




















