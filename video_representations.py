import os
import sys
import time
import configparser
import numpy as np
import pandas as pd
import random as rn
from glob import glob

import tensorflow as tf

from tqdm import tqdm

from parse_config import parse_args

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

from models_c3d import def_c3d_large_video_ae, \
                       def_c3d_small_video_ae, \
                       def_c3d_large_video_classifier, \
                       def_c3d_small_video_classifier2
from models_lstm import def_lstm_large_video_ae, def_lstm_small_video_ae
from models_gru import def_gru_large_video_ae, def_gru_small_video_ae
from models_residual import def_r3d
from models_cnn_lstm import def_cnnlstm_small_video_classifier

from bouncingMNIST import BouncingMNISTDataGenerator
from kth import KTHDataGenerator
from ucf101 import UCF101DataGenerator
from yup import YUPDataGenerator

from utils import plot_metrics, plot_metrics_individually, \
                  get_labels, SVMEval, check_folder

tl = tf.layers

os.environ["PYTHONHASHSEED"] = '42'
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


class VideoRep():

    def __init__(self, sess,
                 model_id, model_name, model_type, ex_mode, model_load_type,
                 dataset_name, class_velocity, hide_digits, step_length,
                 tr_size, val_size,
                 use_batch_norm, use_layer_norm, use_l2_reg, pretrained_cnn,
                 epoch, batch_size, learning_rate,
                 svm_analysis, vis_epoch, plot_individually, verbosity,
                 checkpoint_epoch, keep_checkpoint_max, redo,
                 slack_bot,
                 plt_dir, model_dir, config_dir):
        self.sess = sess

        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.ex_mode = ex_mode
        self.model_load_type = model_load_type

        self.is_ae = (self.model_type.find('ae') != -1)
        self.is_cnn_lstm = (self.model_type.find('cnn_lstm') != -1)

        self.dataset_name = dataset_name
        if self.dataset_name in ['bouncingMNIST']:
            self.is_multilabel = True
        else:
            self.is_multilabel = False
        self.class_velocity = class_velocity
        self.hide_digits = hide_digits
        self.step_length = step_length
        self.tr_size = tr_size
        self.val_size = val_size

        if self.dataset_name in ['bouncingMNIST']:
            self.n_classes = 10              # number of possible classes
            self.n_labels = 2                # number of labels for each sample
        elif self.dataset_name in ['KTH']:
            self.n_classes = 6               # number of possible classes
            self.n_labels = 1                # number of labels for each sample
        elif self.dataset_name in ['UCF101']:
            self.n_classes = 101             # number of possible classes
            self.n_labels = 1                # number of labels for each sample
        elif self.dataset_name in ['YUP']:
            self.n_classes = 20              # number of possible classes
            self.n_labels = 1                # number of labels for each sample

        self.training_cnn = False

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_l2_reg = use_l2_reg
        self.pretrained_cnn = pretrained_cnn

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = 0.9

        self.svm_analysis = svm_analysis
        self.vis_epoch = vis_epoch
        self.plot_individually = plot_individually
        self.verbosity = verbosity

        self.checkpoint_epoch = checkpoint_epoch
        self.keep_checkpoint_max = keep_checkpoint_max
        self.redo = redo

        if slack_bot:
            sys.path.insert(0,
                            '/store/gbpcosta/'
                            'google_drive/PhD/random/slack-bot/')
            from slackbot import SlackBot

            config = configparser.ConfigParser()
            config.read('slack.config')
            self.bot = SlackBot(token=config['SLACK']['token'],
                                channel_name=config['SLACK']['channel_name'])
        else:
            self.bot = None

        self.plt_dir = plt_dir.format(self._get_model_name())
        self.model_dir = model_dir.format(self._get_model_name())
        self.config_dir = config_dir.format(self._get_model_name())

        self._def_model()
        self.current_epoch = 1


        if self.ex_mode == 'train':
            self.tr_digits_only = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    class_velocity=False,
                    hide_digits=False)
            self.val_digits_only = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    split='test',
                    class_velocity=False,
                    hide_digits=False)
            self.tr_digits_only_slow = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=0.03,
                    class_velocity=False,
                    hide_digits=False)
            self.val_digits_only_slow = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=0.03,
                    split='test',
                    class_velocity=False,
                    hide_digits=False)
            self.tr_velocity_only_slow = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=0.03,
                    class_velocity=True,
                    hide_digits=True)
            self.val_velocity_only_slow = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=0.03,
                    split='test',
                    class_velocity=True,
                    hide_digits=True)
            self.tr_velocity_only = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    class_velocity=True,
                    hide_digits=True)
            self.val_velocity_only = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    split='test',
                    class_velocity=True,
                    hide_digits=True)
            self.tr_digits_velocity_slow = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=0.03,
                    class_velocity=True,
                    hide_digits=False)
            self.val_digits_velocity_slow = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=0.03,
                    split='test',
                    class_velocity=True,
                    hide_digits=False)
            self.tr_digits_velocity = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    class_velocity=True,
                    hide_digits=False)
            self.val_digits_velocity = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    split='test',
                    class_velocity=True,
                    hide_digits=False)
            self.tr_KTH = \
                KTHDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='train_validation', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='KTH')
            self.val_KTH = \
                KTHDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='test', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='KTH')
            self.tr_UCF101 = \
                UCF101DataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='train', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='UCF101')
            self.val_UCF101 = \
                UCF101DataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='test', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='UCF101')
            self.tr_YUP = \
                YUPDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='train', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='Yup++')
            self.val_YUP = \
                YUPDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='test', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='Yup++')

        if self.dataset_name == 'bouncingMNIST':
            self.training_generator = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    noise=self.is_ae,
                    step_length=self.step_length,
                    class_velocity=self.class_velocity,
                    hide_digits=self.hide_digits)
            self.validation_generator = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    step_length=self.step_length,
                    split='test',
                    class_velocity=self.class_velocity,
                    hide_digits=self.hide_digits)
        elif self.dataset_name == 'KTH':
            self.training_generator = \
                KTHDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='train_validation', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='KTH')
            self.validation_generator = \
                KTHDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='test', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='KTH')
        elif self.dataset_name == 'UCF101':
            self.training_generator = \
                UCF101DataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='train', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='UCF101')
            self.validation_generator = \
                UCF101DataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='test', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='UCF101')
        elif self.dataset_name == 'YUP':
            self.training_generator = \
                YUPDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='train', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='Yup++')
            self.validation_generator = \
                YUPDataGenerator(
                    seq_length=16, batch_size=32,
                    image_size=64, shuffle=True,
                    frame_skip=1, to_gray=True,
                    split='test', random_seed=42,
                    ae=self.is_ae, noise=False,
                    data_path='Yup++')

        self.num_batches = \
            self.training_generator.dataset_size_ // \
            self.training_generator.batch_size_
        self.val_num_batches = \
            self.validation_generator.dataset_size_ // \
            self.validation_generator.batch_size_

        self.saver = tf.train.Saver(max_to_keep=keep_checkpoint_max)

        self.bestacc_saver = tf.train.Saver(max_to_keep=1)
        self.bestacc = 0

        latest_checkpoint = None
        if not self.redo:
            if self.model_load_type == 'latest':
                latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)

                if latest_checkpoint is not None:
                    if self.verbosity >= 1:
                        print('Loading checkpoint - {}'.format(latest_checkpoint))
                    self.saver.restore(self.sess, latest_checkpoint)

                    if self.is_cnn_lstm:
                        self.training_cnn = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'training_cnn').values[0, 0]
                    else:
                        self.training_cnn = False

                    if self.training_cnn:
                        self.plt_loss = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'cnn_loss').values \
                            .flatten()

                        self.val_loss = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'val_cnn_loss').values \
                            .flatten()

                        self.tr_net_acc = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'tr_cnn_net_acc').values \
                            .flatten()

                        self.val_net_acc = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'val_cnn_net_acc').values \
                            .flatten()

                        if self.svm_analysis:
                            # self.val_auc = pd.read_hdf(
                            #     os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            #     'val_cnn_auc').values

                            self.val_acc = pd.read_hdf(
                                os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                'val_cnn_acc').values
                    else:
                        # Load training losses from previous checkpoint
                        trained_cnn = False
                        try:
                            self.plt_loss = pd.read_hdf(
                                os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                'loss').values \
                                .flatten()

                            # Load validation losses from previous checkpoint
                            self.val_loss = pd.read_hdf(
                                os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                'val_loss').values \
                                .flatten()

                            if self.is_ae is False:
                                self.tr_net_acc = pd.read_hdf(
                                    os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'tr_net_acc').values \
                                    .flatten()

                                self.val_net_acc = pd.read_hdf(
                                    os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'val_net_acc').values \
                                    .flatten()

                            if self.svm_analysis:
                                # self.val_auc = pd.read_hdf(
                                #     os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                #     'val_auc').values

                                self.val_acc = pd.read_hdf(
                                    os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'val_acc').values
                        except KeyError:
                            trained_cnn = True

                    if trained_cnn:
                        self.current_epoch = 1
                    else:
                        self.current_epoch = (self.plt_loss.shape[0]
                                              // self.num_batches) + 1

                    if self.verbosity >= 1:
                        print('Checkpoint loaded - epoch {}'
                              .format(self.current_epoch))
            elif self.model_load_type == 'best':
                latest_checkpoint = sorted(glob(os.path.join(self.model_dir, 'bestacc_model-epoch-*.index')))

                if latest_checkpoint is not None and latest_checkpoint:
                    latest_checkpoint = latest_checkpoint[-1].rsplit('.', 1)[0]
                    print(latest_checkpoint)
                    epoch = int(latest_checkpoint.rsplit('-', 1)[1])

                    if self.verbosity >= 1:
                        print('Loading checkpoint - {}'.
                              format(latest_checkpoint))
                    self.saver.restore(self.sess, latest_checkpoint)

                    if self.is_cnn_lstm:
                        self.training_cnn = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'training_cnn').values[0, 0]
                    else:
                        self.training_cnn = False

                    if self.training_cnn:
                        self.plt_loss = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'cnn_loss').values \
                            .flatten()[:epoch]

                        self.val_loss = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'val_cnn_loss').values \
                            .flatten()[:epoch]

                        self.tr_net_acc = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'tr_cnn_net_acc').values \
                            .flatten()[:epoch]

                        self.val_net_acc = pd.read_hdf(
                            os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            'val_cnn_net_acc').values \
                            .flatten()[:epoch]

                        if self.svm_analysis:
                            # self.val_auc = pd.read_hdf(
                            #     os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                            #     'val_cnn_auc').values

                            self.val_acc = pd.read_hdf(
                                os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                'val_cnn_acc').values[:epoch]
                    else:
                        # Load training losses from previous checkpoint
                        trained_cnn = False
                        try:
                            self.plt_loss = pd.read_hdf(
                                os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                'loss').values \
                                .flatten()[:epoch]

                            # Load validation losses from previous checkpoint
                            self.val_loss = pd.read_hdf(
                                os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                'val_loss').values \
                                .flatten()[:epoch]

                            if self.is_ae is False:
                                self.tr_net_acc = pd.read_hdf(
                                    os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'tr_net_acc').values \
                                    .flatten()[:epoch]

                                self.val_net_acc = pd.read_hdf(
                                    os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'val_net_acc').values \
                                    .flatten()[:epoch]

                            if self.svm_analysis:
                                # self.val_auc = pd.read_hdf(
                                #     os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                                #     'val_auc').values

                                self.val_acc = pd.read_hdf(
                                    os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'val_acc').values[:epoch]
                        except KeyError:
                            trained_cnn = True

                        self.current_epoch = epoch

        if self.current_epoch == 1:
            self.plt_loss = np.array([])
            self.val_loss = np.array([])

            if self.is_ae is False:
                self.tr_net_acc = np.array([])
                self.val_net_acc = np.array([])

            if self.svm_analysis:
                # self.val_auc = np.array([]) \
                #     .reshape(0, self.n_classes)
                self.val_acc = np.array([]) \
                    .reshape(0, self.n_classes)

            if self.is_cnn_lstm and latest_checkpoint is None:
                self.training_cnn = True

    def _get_model_name(self):
        # return 'semantic_nn_{}_ae_{}_{}emb_{}epoch_{}' \
        #     .format(self.comb_type,
        #             self.ae_type,
        #             self.emb_dim,
        #             self.epoch,
        #             model_name)
        return '{}_{}'.format(self.model_name, self.model_id)

    def _def_loss_fn(self):
        if self.is_ae:  # mse
            self.loss = tf.reduce_mean(
                tf.square(self.net_out - self.video))

        else:  # binary_crossentropy
            if self.is_cnn_lstm:
                if self.is_multilabel:
                    if not self.pretrained_cnn:
                        self.cnn_loss = tf.reduce_mean(tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.cnn_net_out_logits,
                                labels=self.frame_labels), axis=1))

                    self.loss = tf.reduce_mean(tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.net_out_logits,
                            labels=self.frame_labels), axis=1))
                else:
                    if not self.pretrained_cnn:
                        self.cnn_loss = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.cnn_net_out_logits,
                                labels=self.frame_labels))

                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.net_out_logits,
                            labels=self.frame_labels))
            else:
                if self.is_multilabel:
                    self.loss = tf.reduce_mean(tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.net_out_logits,
                            labels=self.labels), axis=1))
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=self.net_out_logits,
                            labels=self.labels))

        if self.use_batch_norm:
            if self.is_ae:  # mse
                self.test_loss = tf.reduce_mean(
                    tf.square(self.test_net_out - self.video))

            else:  # binary_crossentropy
                if self.is_cnn_lstm:
                    if self.is_multilabel:
                        if not self.pretrained_cnn:
                            self.test_cnn_loss = tf.reduce_mean(tf.reduce_sum(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=self.test_cnn_net_out_logits,
                                    labels=self.labels), axis=1))

                        self.test_loss = tf.reduce_mean(tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.test_net_out_logits,
                                labels=self.labels), axis=1))
                    else:
                        if not self.pretrained_cnn:
                            self.test_cnn_loss = tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                    logits=self.test_cnn_net_out_logits,
                                    labels=self.labels))

                        self.test_loss = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.test_net_out_logits,
                                labels=self.labels))
                else:
                    if self.is_multilabel:
                        self.test_loss = tf.reduce_mean(tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.test_net_out_logits,
                                labels=self.labels), axis=1))
                    else:
                        self.test_loss = tf.reduce_mean(
                            tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=self.test_net_out_logits,
                                labels=self.labels))

    def _def_optimizer(self):
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.is_cnn_lstm:
                if not self.pretrained_cnn:
                    self.cnn_optim = \
                        tf.train.AdamOptimizer(learning_rate=5e-4,  # self.learning_rate,
                                               beta1=self.beta1) \
                        .minimize(self.cnn_loss,
                                  var_list=self.cnn_vars)
                self.optim = \
                    tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=self.beta1) \
                    .minimize(self.loss,
                              var_list=self.lstm_vars)
            else:
                self.optim = \
                    tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=self.beta1) \
                    .minimize(self.loss,
                              var_list=self.learnable_vars)

    def _def_svm_clf(self):
        self.svm = SVMEval(tr_size=self.tr_size, val_size=self.val_size,
                           n_splits=5, scale=False, per_class=True,
                           n_labels=self.n_labels, verbose=self.verbosity)

    def _def_eval_metrics(self):
        if self.is_ae:
            pass
        else:
            if self.is_multilabel:
                labels = tf.squeeze(tf.nn.top_k(self.labels).indices)
                predictions = tf.squeeze(tf.nn.top_k(self.net_out).indices)

                self.acc, self.acc_op = tf.metrics.mean_per_class_accuracy(
                        labels=labels,
                        predictions=predictions,
                        num_classes=10, name='net_acc')
            else:
                self.acc, self.acc_op = tf.metrics.accuracy(
                    labels=tf.argmax(self.labels, axis=1),
                    predictions=tf.argmax(self.net_out_logits, axis=1),
                    name='net_acc')
        eval_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                      scope="net_acc")
        init_eval_vars = tf.variables_initializer(var_list=eval_vars)

        return init_eval_vars

    def _def_model(self):
        self.video = tf.placeholder(tf.float32, [None, 16, 64, 64, 1])

        if self.is_multilabel:
            self.labels = tf.placeholder(tf.float32, [None, self.n_classes, 2])
        else:
            self.labels = tf.placeholder(tf.float32, [None, self.n_classes])

        if self.is_multilabel:
            self.get_onehot_labels = \
                tf.squeeze(tf.nn.top_k(self.labels).indices)
        else:
            self.get_onehot_labels = self.labels

        if self.is_cnn_lstm:
            if self.pretrained_cnn:
                pass

            if self.is_multilabel:
                frame_labels = tf.expand_dims(self.labels, axis=1)

                frame_labels = \
                    tf.tile(frame_labels,
                            [1, self.video.get_shape().as_list()[1], 1, 1])
                self.frame_labels = tf.reshape(frame_labels,
                                               [-1, self.n_classes, 2])
                # self.frame_labels = tf.cast(frame_labels, tf.float32)

            else:
                frame_labels = tf.expand_dims(self.labels, axis=1)
                frame_labels = \
                    tf.tile(frame_labels,
                            [1, self.video.get_shape().as_list()[1], 1])
                self.frame_labels = tf.reshape(frame_labels,
                                               [-1, self.n_classes])
                # self.frame_labels = tf.cast(frame_labels, tf.float32)

        if self.model_type == 'c3d_ae_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_c3d_small_video_ae(
                    self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'c3d_clf_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_c3d_small_video_classifier2(
                    self.video,  n_classes=self.n_classes,
                    is_training=True, is_multilabel=self.is_multilabel,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'c3d_ae_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_c3d_large_video_ae(
                    input=self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'c3d_clf_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_c3d_large_video_classifier(
                    self.video, n_classes=self.n_classes,
                    is_training=True, is_multilabel=self.is_multilabel,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'lstm_ae_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_lstm_small_video_ae(
                    self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'lstm_ae_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_lstm_large_video_ae(
                    self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'cnn_lstm_clf_small':
            (self.cnn_net_out, self.net_out), \
                (self.cnn_net_out_logits, self.net_out_logits), \
                (self.cnn_feats, self.video_emb), \
                (self.cnn_vars, self.lstm_vars), \
                (self.cnn_emb_dim, self.emb_dim), \
                self.cnn_model = \
                def_cnnlstm_small_video_classifier(
                    self.video, n_classes=self.n_classes,
                    is_training=True, is_multilabel=self.is_multilabel,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm,
                    pretrained_cnn=self.pretrained_cnn)
        elif self.model_type == 'gru_ae_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_gru_small_video_ae(
                    self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'gru_ae_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_gru_large_video_ae(
                    self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'r3d_clf_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_r3d(input=self.video, n_classes=self.n_classes,
                        is_training=True,  is_multilabel=self.is_multilabel,
                        model_depth=10, model_size='small',
                        is_decomposed=False, verbosity=self.verbosity,
                        reuse=False, video_emb_layer_name='res3')
        elif self.model_type == 'r3d_clf_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_r3d(input=self.video, n_classes=self.n_classes,
                        is_training=True, is_multilabel=self.is_multilabel,
                        model_depth=18, model_size='small',
                        is_decomposed=False, verbosity=self.verbosity,
                        reuse=False, video_emb_layer_name='res7')
        elif self.model_type == 'r21d_clf_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_r3d(input=self.video, n_classes=self.n_classes,
                        is_training=True,  is_multilabel=self.is_multilabel,
                        model_depth=10, model_size='small',
                        is_decomposed=True, verbosity=self.verbosity,
                        reuse=False, video_emb_layer_name='res3')
        elif self.model_type == 'r21d_clf_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_r3d(input=self.video, n_classes=self.n_classes,
                        is_training=True, is_multilabel=self.is_multilabel,
                        model_depth=18, model_size='small',
                        is_decomposed=True, verbosity=self.verbosity,
                        reuse=False, video_emb_layer_name='res7')

        if self.is_multilabel:
            self.get_onehot_pred = \
                tf.squeeze(tf.nn.top_k(self.net_out).indices)
        else:
            self.get_onehot_pred = \
                tf.one_hot(tf.argmax(self.net_out, axis=1), self.n_classes)

        if self.model_type.find('cnn_lstm') != -1:
            if self.is_multilabel:
                self.get_onehot_cnn_pred = \
                    tf.squeeze(tf.nn.top_k(self.cnn_net_out).indices)
            else:
                self.get_onehot_cnn_pred = \
                    tf.one_hot(tf.argmax(self.cnn_net_out, axis=1),
                               self.n_classes)

        if self.use_batch_norm:
            if self.model_type == 'c3d_ae_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_c3d_small_video_ae(self.video,
                                           is_training=False,
                                           reuse=True,
                                           use_l2_reg=self.use_l2_reg,
                                           use_batch_norm=self.use_batch_norm,
                                           use_layer_norm=self.use_layer_norm)
            elif self.model_type == 'c3d_clf_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_c3d_small_video_classifier2(
                        self.video, n_classes=self.n_classes,
                        is_training=False, is_multilabel=self.is_multilabel,
                        reuse=True,
                        use_l2_reg=self.use_l2_reg,
                        use_batch_norm=self.use_batch_norm,
                        use_layer_norm=self.use_layer_norm)
            elif self.model_type == 'c3d_ae_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_c3d_large_video_ae(self.video,
                                           is_training=False,
                                           reuse=True)
            elif self.model_type == 'c3d_clf_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_c3d_large_video_classifier(
                        self.video, n_classes=self.n_classes,
                        is_training=False, is_multilabel=self.is_multilabel,
                        reuse=True,
                        use_l2_reg=self.use_l2_reg,
                        use_batch_norm=self.use_batch_norm,
                        use_layer_norm=self.use_layer_norm)
            elif self.model_type == 'lstm_ae_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_lstm_small_video_ae(self.video,
                                            is_training=False,
                                            reuse=True)
            elif self.model_type == 'lstm_ae_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_lstm_large_video_ae(self.video,
                                            is_training=False,
                                            reuse=True)
            elif self.model_type == 'cnn_lstm_clf_small':
                (self.test_cnn_net_out, self.test_net_out), \
                    (self.test_cnn_net_out_logits,
                     self.test_net_out_logits),\
                    (self.test_cnn_feats, self.test_video_emb), \
                    (self.test_cnn_vars, self.test_lstm_vars), \
                    (self.cnn_emb_dim, self.emb_dim), \
                    self.cnn_model = \
                    def_cnnlstm_small_video_classifier(
                        self.video, n_classes=self.n_classes,
                        is_training=False, is_multilabel=self.is_multilabel,
                        reuse=True,
                        use_l2_reg=self.use_l2_reg,
                        use_batch_norm=self.use_batch_norm,
                        use_layer_norm=self.use_layer_norm,
                        pretrained_cnn=self.pretrained_cnn)
            elif self.model_type == 'gru_ae_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_gru_small_video_ae(self.video,
                                           is_training=False,
                                           reuse=True)
            elif self.model_type == 'gru_ae_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_gru_large_video_ae(self.video,
                                           is_training=False,
                                           reuse=True)
            elif self.model_type == 'r3d_clf_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_r3d(input=self.video, n_classes=self.n_classes,
                            is_training=False,
                            is_multilabel=self.is_multilabel,
                            model_depth=10, model_size='small',
                            is_decomposed=False, verbosity=self.verbosity,
                            reuse=True, video_emb_layer_name='res3')
            elif self.model_type == 'r3d_clf_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_r3d(input=self.video, n_classes=self.n_classes,
                            is_training=False,
                            is_multilabel=self.is_multilabel,
                            model_depth=18, model_size='small',
                            is_decomposed=False, verbosity=self.verbosity,
                            reuse=True, video_emb_layer_name='res7')
            elif self.model_type == 'r21d_clf_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_r3d(input=self.video, n_classes=self.n_classes,
                            is_training=False,
                            is_multilabel=self.is_multilabel,
                            model_depth=10, model_size='small',
                            is_decomposed=True, verbosity=self.verbosity,
                            reuse=True, video_emb_layer_name='res3')
            elif self.model_type == 'r21d_clf_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_r3d(input=self.video, n_classes=self.n_classes,
                            is_training=False,
                            is_multilabel=self.is_multilabel,
                            model_depth=18, model_size='small',
                            is_decomposed=True, verbosity=self.verbosity,
                            reuse=True, video_emb_layer_name='res7')

            if self.is_multilabel:
                self.get_onehot_test_pred = \
                    tf.squeeze(tf.nn.top_k(self.test_net_out).indices)
            else:
                self.get_onehot_test_pred = \
                    tf.one_hot(tf.argmax(self.test_net_out, axis=1),
                               self.n_classes)

            if self.model_type.find('cnn_lstm') != -1:
                if self.is_multilabel:
                    self.get_onehot_test_cnn_pred = \
                        tf.squeeze(tf.nn.top_k(self.test_cnn_net_out).indices)
                else:
                    self.get_onehot_test_cnn_pred = \
                        tf.one_hot(tf.argmax(self.test_cnn_net_out, axis=1),
                                   self.n_classes)

        self._def_loss_fn()
        self._def_optimizer()
        if self.svm_analysis:
            self._def_svm_clf()

    def train_cnn(self):
        start_time = time.time()

        starting_epoch = self.current_epoch

        self.training_cnn = True

        for epoch in tqdm(range(starting_epoch, self.epoch+1), position=1):

            ###################################################################
            # Training
            ###################################################################

            for batch_number in tqdm(range(1, self.num_batches+1),
                                     position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                _, loss = \
                    self.sess.run([self.cnn_optim,
                                   self.cnn_loss],
                                  feed_dict={self.video: video_batch,
                                             self.labels: labels})

                if self.verbosity >= 1:
                    print('CNN Training - Epoch: [%2d] [%4d / %4d] time: '
                          '%4.4f, loss: %.8f'
                          % (epoch, batch_number, self.num_batches,
                             time.time() - start_time,
                             loss))

                self.plt_loss = np.append(self.plt_loss, loss)

            self.training_generator.on_epoch_end()

            ###################################################################
            # Training Evaluation
            ###################################################################

            tr_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.cnn_emb_dim)
            tr_labels = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)
            tr_pred = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)

            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm:
                    video_emb, labels, pred = \
                        self.sess.run(
                            [self.test_cnn_feats,
                             self.get_onehot_labels,
                             self.get_onehot_test_cnn_pred],
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})
                else:
                    video_emb, labels, pred = \
                        self.sess.run(
                            [self.cnn_feats,
                             self.get_onehot_labels,
                             self.get_onehot_cnn_pred],
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})

                tr_video_emb = np.vstack([tr_video_emb, video_emb[15::16]])
                tr_labels = np.vstack([tr_labels, labels])

                if self.is_ae is False:
                    tr_pred = np.vstack([tr_pred, pred])

            if self.is_ae is False:
                self.tr_net_acc = np.append(
                    self.tr_net_acc,
                    accuracy_score(tr_labels, tr_pred))

            self.training_generator.on_epoch_end()

            ###################################################################
            # Best Accuracy Checkpoint
            ###################################################################

            if self.bestacc < self.tr_net_acc[-1]:
                self.bestacc = self.tr_net_acc[-1]
                self.bestacc_saver.save(
                    self.sess,
                    os.path.join(
                        self.model_dir,
                        'cnn_bestacc_model-epoch'),
                    global_step=epoch)

            ###################################################################
            # Validation
            ###################################################################

            val_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.cnn_emb_dim)

            val_labels = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)
            val_pred = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)

            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.validation_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm:
                    loss, net_out, video_emb, labels, pred = \
                        self.sess.run(
                            [self.test_cnn_loss,
                             self.test_cnn_net_out,
                             self.test_cnn_feats,
                             self.get_onehot_labels,
                             self.get_onehot_test_cnn_pred],
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})
                else:
                    loss, net_out, video_emb, labels, pred = \
                        self.sess.run(
                            [self.cnn_loss,
                             self.cnn_net_out,
                             self.cnn_feats,
                             self.get_onehot_labels,
                             self.get_onehot_test_cnn_pred],
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})

                self.val_loss = np.append(self.val_loss, loss)
                val_video_emb = np.vstack([val_video_emb, video_emb[15::16]])
                val_labels = np.vstack([val_labels, labels])

                if self.is_ae is False:
                    val_pred = np.vstack([val_pred, pred])

            if self.is_ae is False:
                self.val_net_acc = np.append(
                    self.val_net_acc,
                    accuracy_score(val_labels, val_pred))

            self.validation_generator.on_epoch_end()

            ###################################################################
            # Visualisation
            ###################################################################

            if epoch % self.vis_epoch == 0 or \
                    epoch == 1 or \
                    epoch == self.epoch:

                if self.svm_analysis:
                    # val_auc,
                    # tr_labels = np.expand_dims(tr_labels, axis=1)
                    # tr_labels = \
                    #     np.tile(tr_labels,
                    #             [1, video_batch.shape[1], 1])
                    # tr_labels = tr_labels.reshape([-1, self.n_classes])

                    # svm_val_labels = np.expand_dims(val_labels, axis=1)
                    # svm_val_labels = \
                    #     np.tile(svm_val_labels,
                    #             [1, video_batch.shape[1], 1])
                    # svm_val_labels = svm_val_labels.reshape([-1, self.n_classes])

                    val_acc, val_svm_pred, valid_classes = \
                        self.svm.compute(train_data=(tr_video_emb, tr_labels),
                                         val_data=(val_video_emb, val_labels))
                    self.val_acc = np.vstack([self.val_acc, val_acc])
                    # self.val_auc = np.vstack([self.val_auc, val_auc])

                    plt_acc = []
                    # plt_auc = []
                    metrics_it_list = []
                    acc_plt_names = []
                    # auc_plt_names = []
                    metrics_plt_types = []
                    for cl in range(self.n_classes):
                        plt_acc.append(self.val_acc[:, cl])
                        # plt_auc.append(self.val_auc[:, cl])
                        if self.vis_epoch == 1:
                            metrics_it_list.append(
                                list(range(self.vis_epoch,
                                           self.current_epoch+1,
                                           self.vis_epoch)))
                        elif epoch != self.epoch:
                            metrics_it_list.append(
                                [1]
                                + list(range(self.vis_epoch,
                                             self.current_epoch+1,
                                             self.vis_epoch)))
                        elif epoch == self.epoch and \
                                epoch % self.vis_epoch != 0:
                            metrics_it_list.append(
                                [1]
                                + list(range(self.vis_epoch,
                                             self.current_epoch+1,
                                             self.vis_epoch))
                                + [epoch])
                        else:
                            metrics_it_list.append(
                                [1]
                                + list(range(self.vis_epoch,
                                             self.current_epoch+1,
                                             self.vis_epoch)))
                        metrics_plt_types.append('lines')
                        acc_plt_names.append('Validation Accuracy (SVM) - '
                                             'Class {}'.format(cl))
                        # auc_plt_names.append('Validation ROC AUC (SVM) - '
                        #                      'Class {}'.format(cl))


                plt_idx = \
                    np.random.choice(
                        range(len(val_video_emb)),
                        size=self.validation_generator.dataset_size_)
                plt_val_video_emb = val_video_emb[plt_idx]

                val_labels_n = get_labels(val_labels, n_labels=self.n_labels)
                val_labels_n = np.expand_dims(val_labels_n, axis=1)

                val_labels_n = \
                    np.tile(val_labels_n,
                            [1, video_batch.shape[1], 1])
                val_labels_n = val_labels_n.reshape([-1, self.n_labels])

                pca_projs = []
                lda_projs = []
                tsne_projs = []
                for ii in range(self.n_labels):
                    label = val_labels_n[plt_idx, ii].reshape((-1, 1))

                    # PCA
                    if ii == 0:
                        pca = PCA(n_components=2, whiten=True)
                        pca_video_proj = pca.fit_transform(plt_val_video_emb)

                    pca_video_proj_aux = np.concatenate([pca_video_proj,
                                                         label],
                                                        axis=1)
                    pca_projs.append(pca_video_proj_aux)

                    # LDA
                    lda = LinearDiscriminantAnalysis(n_components=2)
                    lda_video_proj = lda.fit_transform(plt_val_video_emb,
                                                       label.squeeze())
                    lda_video_proj = np.concatenate([lda_video_proj, label],
                                                    axis=1)
                    lda_projs.append(lda_video_proj)

                    # t-SNE
                    if ii == 0:
                        tsne = TSNE(n_components=2)
                        tsne_video_proj = tsne.fit_transform(plt_val_video_emb)
                        tsne_video_proj = \
                            tsne_video_proj[:self.validation_generator.dataset_size_]

                    tsne_video_proj_aux = np.concatenate([tsne_video_proj,
                                                          label],
                                                         axis=1)
                    tsne_projs.append(tsne_video_proj_aux)

                if self.is_ae:
                    metrics_list = [self.plt_loss] \
                                   + pca_projs \
                                   + lda_projs \
                                   + tsne_projs \
                                   + [video_batch[0].squeeze(),
                                      net_out[0].squeeze(),
                                      video_batch[1].squeeze(),
                                      net_out[1].squeeze()]
                    iterations_list = [list(range(1, (self.current_epoch
                                                      * self.num_batches)+1))]\
                        + self.n_labels * [None] \
                        + self.n_labels * [None] \
                        + self.n_labels * [None] \
                        + [None, None, None, None]
                    plt_types = ['lines'] \
                        + self.n_labels * ['scatter'] \
                        + self.n_labels * ['scatter'] \
                        + self.n_labels * ['scatter'] \
                        + ['image-grid', 'image-grid',
                           'image-grid', 'image-grid']
                    metric_names = ['Loss'] \
                        + ['PCA Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['LDA Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['t-SNE Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['Video 1',
                           'Video Reconstruction AE 1',
                           'Video 2',
                           'Video Reconstruction AE 2']

                else:
                    metrics_list = [self.plt_loss,
                                    self.tr_net_acc,
                                    self.val_net_acc] \
                                   + pca_projs \
                                   + lda_projs \
                                   + tsne_projs
                    iterations_list = [list(range(1, (self.current_epoch
                                                      * self.num_batches)+1)),
                                       list(range(1, self.current_epoch+1)),
                                       list(range(1, self.current_epoch+1))] \
                        + self.n_labels * [None] \
                        + self.n_labels * [None] \
                        + self.n_labels * [None]
                    plt_types = ['lines', 'lines', 'lines'] \
                        + self.n_labels * ['scatter'] \
                        + self.n_labels * ['scatter'] \
                        + self.n_labels * ['scatter']
                    metric_names = ['Loss',
                                    'Training Network Accuracy',
                                    'Validation Network Accuracy'] \
                        + ['PCA Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['LDA Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['t-SNE Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)]

                plot_metrics(
                    metrics_list=metrics_list,
                    iterations_list=iterations_list,
                    types=plt_types,
                    metric_names=metric_names,
                    legend=True,
                    # x_label='Iteration',
                    # y_label='Loss',
                    savefile=os.path.join(self.plt_dir,
                                          'cnn_metrics_epoch{}.png'
                                          .format(epoch)),
                    bot=self.bot)

                if self.plot_individually:
                    plot_metrics_individually(
                        metrics_list=metrics_list,
                        iterations_list=iterations_list,
                        types=plt_types,
                        metric_names=metric_names,
                        legend=True,
                        # x_label='Iteration',
                        # y_label='Loss',
                        savefile=os.path.join(self.plt_dir,
                                              'cnn_metrics_epoch{}.png'
                                              .format(epoch)),
                        bot=None)

                if self.svm_analysis:
                    plot_metrics(
                        metrics_list=plt_acc,
                        iterations_list=metrics_it_list,
                        types=metrics_plt_types,
                        metric_names=acc_plt_names,
                        legend=True,
                        # x_label='Iteration',
                        # y_label='Loss',
                        savefile=os.path.join(self.plt_dir,
                                              'cnn_acc_epoch{}.png'
                                              .format(epoch)),
                        bot=self.bot)
                    if self.plot_individually:
                        plot_metrics_individually(
                            metrics_list=plt_acc,
                            iterations_list=metrics_it_list,
                            types=metrics_plt_types,
                            metric_names=acc_plt_names,
                            legend=True,
                            # x_label='Iteration',
                            # y_label='Loss',
                            savefile=os.path.join(self.plt_dir,
                                                  'cnn_acc_epoch{}.png'
                                                  .format(epoch)),
                            bot=None)

                    # plot_metrics(
                    #     metrics_list=plt_auc,
                    #     iterations_list=metrics_it_list,
                    #     types=metrics_plt_types,
                    #     metric_names=auc_plt_names,
                    #     legend=True,
                    #     # x_label='Iteration',
                    #     # y_label='Loss',
                    #     savefile=os.path.join(self.plt_dir,
                    #                           'auc_epoch{}.png'
                    #                           .format(epoch)),
                    #     bot=self.bot)
                    #
                    # if self.plot_individually:
                    #     plot_metrics_individually(
                    #         metrics_list=plt_auc,
                    #         iterations_list=metrics_it_list,
                    #         types=metrics_plt_types,
                    #         metric_names=auc_plt_names,
                    #         legend=True,
                    #         # x_label='Iteration',
                    #         # y_label='Loss',
                    #         savefile=os.path.join(self.plt_dir,
                    #                               'auc_epoch{}.png'
                    #                               .format(epoch)),
                    #         bot=None)

                confusion_matrices = []
                confusion_matrices_names = []
                for ii in range(self.n_classes):
                    confusion_matrices.append(
                        [confusion_matrix(val_labels[:, ii],
                                          val_svm_pred[:, ii]),
                         [-1, ii]])
                    confusion_matrices_names.append(
                        'Confusion Matrix - Class {}'.format(ii))

                plot_metrics(
                    metrics_list=confusion_matrices,
                    iterations_list=[None] * self.n_classes,
                    types=['confusion-matrix'] * self.n_classes,
                    metric_names=confusion_matrices_names,
                    legend=True,
                    savefile=os.path.join(self.plt_dir,
                                          'cnn_cm_epoch{}.png'
                                          .format(epoch)),
                    bot=self.bot)

                if self.plot_individually:
                    plot_metrics_individually(
                        metrics_list=confusion_matrices,
                        iterations_list=[None] * self.n_classes,
                        types=['confusion-matrix'] * self.n_classes,
                        metric_names=confusion_matrices_names,
                        legend=True,
                        savefile=os.path.join(self.plt_dir,
                                              'cnn_cm_epoch{}.png'
                                              .format(epoch)),
                        bot=None)

            ##################################################################
            # Checkpoints
            ##################################################################

            if epoch % self.checkpoint_epoch == 0 or \
                    epoch == self.epoch:
                n_epochs = self.checkpoint_epoch

                pd.DataFrame([self.training_cnn]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'training_cnn',
                            append=False)

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.plt_loss[-ii*self.num_batches:
                                      -(ii-1)*self.num_batches]
                        .reshape((1, -1)), index=[epoch-ii+1]) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'cnn_loss',
                                append=True, format='table')

                pd.DataFrame(
                    self.plt_loss[-self.num_batches:]
                    .reshape((1, -1)), index=[epoch]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'cnn_loss',
                            append=True, format='table')

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.val_loss[-ii*self.val_num_batches:
                                      -(ii-1)*self.val_num_batches]
                        .reshape((1, -1)), index=[epoch-ii+1]) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'val_cnn_loss',
                                append=True, format='table')

                pd.DataFrame(
                    self.val_loss[-self.val_num_batches:]
                    .reshape((1, -1)), index=[epoch]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_cnn_loss',
                            append=True, format='table')

                if self.is_ae is False:
                    pd.DataFrame(
                        self.tr_net_acc[-self.checkpoint_epoch:],
                        index=list(range(max(1, epoch-self.checkpoint_epoch+1),
                                         epoch+1))) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'tr_cnn_net_acc',
                                append=True, format='table')

                    pd.DataFrame(
                        self.val_net_acc[-self.checkpoint_epoch:],
                        index=list(range(max(1, epoch-self.checkpoint_epoch+1),
                                         epoch+1))) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'val_cnn_net_acc',
                                append=True, format='table')

                if self.svm_analysis:
                    pd.DataFrame(
                        self.val_acc[-1].reshape((1, -1)), index=[epoch]) \
                        .to_hdf(
                            os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_cnn_acc',
                            append=True, format='table')

                    # pd.DataFrame(
                    #     self.val_auc[-1].reshape((1, -1)), index=[epoch]) \
                    #     .to_hdf(
                    #         os.path.join(self.model_dir,
                    #                      'plt_loss_bkup.h5'),
                    #         'val_auc',
                    #         append=True, format='table')

                self.saver.save(
                    self.sess,
                    os.path.join(
                        self.model_dir,
                        'checkpoint-epoch'),
                    global_step=epoch)

            self.training_generator.on_epoch_end()
            self.current_epoch += 1

    def train_model(self):
        if self.current_epoch == 1:
            self.sess.run(tf.global_variables_initializer())

        start_time = time.time()

        starting_epoch = self.current_epoch

        if self.is_cnn_lstm and self.training_cnn:
            # train cnn first
            if self.pretrained_cnn.find('mobilenet') != -1:
                self.sess.run(self.cnn_model.pretrained())
            elif self.pretrained_cnn.find('mnist') != -1:
                self.cnn_model.restore(
                    self.sess,
                    tf.train.latest_checkpoint(
                        os.path.dirname('pretrained_models/mnist_model'
                                        '/model.ckpt-24000.meta')))
            elif not self.pretrained_cnn:
                if self.verbosity >= 1:
                    print('Training CNN first!')
                self.train_cnn()
            else:
                raise NotImplementedError

            self.training_cnn = False
            self.current_epoch = 1

        self.plt_loss = np.array([])
        self.val_loss = np.array([])
        self.tr_net_acc = np.array([])
        self.val_net_acc = np.array([])
        self.val_acc = np.array([]).reshape(0, self.n_classes)
        self.bestacc = 0

        pd.DataFrame([self.training_cnn]) \
            .to_hdf(os.path.join(self.model_dir,
                                 'plt_loss_bkup.h5'),
                    'training_cnn',
                    append=False)

        for epoch in tqdm(range(starting_epoch, self.epoch+1), position=1):

            ###################################################################
            # Training
            ###################################################################

            for batch_number in tqdm(range(1, self.num_batches+1),
                                     position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                _, loss = \
                    self.sess.run([self.optim,
                                   self.loss],
                                  feed_dict={self.video: video_batch,
                                             self.labels: labels})

                if self.verbosity >= 1:
                    print('Epoch: [%2d] [%4d / %4d] time: %4.4f,'
                          ' loss: %.8f'
                          % (epoch, batch_number, self.num_batches,
                             time.time() - start_time,
                             loss))

                self.plt_loss = np.append(self.plt_loss, loss)

            self.training_generator.on_epoch_end()

            ###################################################################
            # Training evaluation
            ###################################################################

            tr_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)
            tr_labels = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)
            tr_pred = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)

            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.is_ae:
                    if self.use_batch_norm:
                        labels, video_emb = \
                            self.sess.run(
                                [self.get_onehot_labels,
                                 self.test_video_emb],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        labels, video_emb = \
                            self.sess.run(
                                [self.get_onehot_labels,
                                 self.video_emb],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                else:
                    if self.use_batch_norm:
                        labels, pred, video_emb = \
                            self.sess.run(
                                [self.get_onehot_labels,
                                 self.get_onehot_test_pred,
                                 self.test_video_emb],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        labels, pred, video_emb = \
                            self.sess.run(
                                [self.get_onehot_labels,
                                 self.get_onehot_pred,
                                 self.video_emb],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})

                tr_video_emb = np.vstack([tr_video_emb, video_emb])
                tr_labels = np.vstack([tr_labels, labels])

                if self.is_ae is False:
                    tr_pred = np.vstack([tr_pred, pred])

            if self.is_ae is False:
                self.tr_net_acc = np.append(
                    self.tr_net_acc,
                    accuracy_score(tr_labels, tr_pred))

            self.training_generator.on_epoch_end()

            ###################################################################
            # Best Accuracy Checkpoint
            ###################################################################

            if self.bestacc < self.tr_net_acc[-1]:
                self.bestacc = self.tr_net_acc[-1]
                self.bestacc_saver.save(
                    self.sess,
                    os.path.join(
                        self.model_dir,
                        'bestacc_model-epoch'),
                    global_step=epoch)

            ###################################################################
            # Validation
            ###################################################################

            val_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)
            val_labels = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)
            val_pred = np.array([], dtype=np.float32) \
                .reshape(0, self.n_classes)

            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.validation_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.is_ae:
                    if self.use_batch_norm:
                        loss, video_emb, labels = \
                            self.sess.run(
                                [self.test_loss,
                                 self.test_video_emb,
                                 self.get_onehot_labels],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        loss, video_emb, labels = \
                            self.sess.run(
                                [self.loss,
                                 self.video_emb,
                                 self.get_onehot_labels],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                else:
                    if self.use_batch_norm:
                        loss, video_emb, labels, pred = \
                            self.sess.run(
                                [self.test_loss,
                                 self.test_video_emb,
                                 self.get_onehot_labels,
                                 self.get_onehot_test_pred],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        loss, video_emb, labels, pred = \
                            self.sess.run(
                                [self.loss,
                                 self.video_emb,
                                 self.get_onehot_labels,
                                 self.get_onehot_pred],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})

                self.val_loss = np.append(self.val_loss, loss)
                val_video_emb = np.vstack([val_video_emb, video_emb])
                val_labels = np.vstack([val_labels, labels])

                if self.is_ae is False:
                    val_pred = np.vstack([val_pred, pred])

            if self.is_ae is False:
                self.val_net_acc = np.append(
                    self.val_net_acc,
                    accuracy_score(val_labels, val_pred))

            self.validation_generator.on_epoch_end()

            ##################################################################
            # Visualisation
            ##################################################################

            if epoch % self.vis_epoch == 0 or \
                    epoch == 1 or \
                    epoch == self.epoch:

                if self.svm_analysis:
                    # val_auc,
                    val_acc, val_svm_pred, valid_classes = \
                        self.svm.compute(train_data=(tr_video_emb, tr_labels),
                                         val_data=(val_video_emb, val_labels))
                    self.val_acc = np.vstack([self.val_acc, val_acc])
                    # self.val_auc = np.vstack([self.val_auc, val_auc])

                    plt_acc = []
                    # plt_auc = []
                    metrics_it_list = []
                    acc_plt_names = []
                    # auc_plt_names = []
                    metrics_plt_types = []
                    for cl in range(self.n_classes):
                        plt_acc.append(self.val_acc[:, cl])
                        # plt_auc.append(self.val_auc[:, cl])
                        if self.vis_epoch == 1:
                            metrics_it_list.append(
                                list(range(self.vis_epoch,
                                           self.current_epoch+1,
                                           self.vis_epoch)))
                        elif epoch != self.epoch:
                            metrics_it_list.append(
                                [1]
                                + list(range(self.vis_epoch,
                                             self.current_epoch+1,
                                             self.vis_epoch)))
                        elif epoch == self.epoch and \
                                epoch % self.vis_epoch != 0:
                            metrics_it_list.append(
                                [1]
                                + list(range(self.vis_epoch,
                                             self.current_epoch+1,
                                             self.vis_epoch))
                                + [epoch])
                        else:
                            metrics_it_list.append(
                                [1]
                                + list(range(self.vis_epoch,
                                             self.current_epoch+1,
                                             self.vis_epoch)))
                        metrics_plt_types.append('lines')
                        acc_plt_names.append('Validation Accuracy (SVM) - '
                                             'Class {}'.format(cl))
                        # auc_plt_names.append('Validation ROC AUC (SVM) - '
                        #                      'Class {}'.format(cl))

                val_labels_n = get_labels(val_labels, n_labels=self.n_labels)
                pca_projs = []
                lda_projs = []
                tsne_projs = []
                for ii in range(self.n_labels):
                    label = val_labels_n[:, ii].reshape((-1, 1))

                    # PCA
                    if ii == 0:
                        pca = PCA(n_components=2, whiten=True)
                        pca_video_proj = pca.fit_transform(val_video_emb)

                    pca_video_proj_aux = np.concatenate([pca_video_proj,
                                                         label],
                                                        axis=1)
                    pca_projs.append(pca_video_proj_aux)

                    # LDA
                    lda = LinearDiscriminantAnalysis(n_components=2)
                    lda_video_proj = lda.fit_transform(val_video_emb,
                                                       label.squeeze())
                    lda_video_proj = np.concatenate([lda_video_proj, label],
                                                    axis=1)
                    lda_projs.append(lda_video_proj)

                    # t-SNE
                    if ii == 0:
                        tsne = TSNE(n_components=2)
                        tsne_video_proj = tsne.fit_transform(val_video_emb)
                        tsne_video_proj = \
                            tsne_video_proj[:self.validation_generator.dataset_size_]
                    tsne_video_proj_aux = np.concatenate([tsne_video_proj,
                                                          label],
                                                         axis=1)
                    tsne_projs.append(tsne_video_proj_aux)

                if self.is_ae:
                    metrics_list = [self.plt_loss] \
                                   + pca_projs \
                                   + lda_projs \
                                   + tsne_projs \
                                   + [video_batch[0].squeeze(),
                                      net_out[0].squeeze(),
                                      video_batch[1].squeeze(),
                                      net_out[1].squeeze()]
                    iterations_list = [list(range(1, (self.current_epoch
                                                      * self.num_batches)+1))] \
                                      + self.n_labels * [None] \
                                      + self.n_labels * [None] \
                                      + self.n_labels * [None] \
                                      + [None, None, None, None]
                    plt_types = ['lines'] \
                                + self.n_labels * ['scatter'] \
                                + self.n_labels * ['scatter'] \
                                + self.n_labels * ['scatter'] \
                                + ['image-grid', 'image-grid',
                                   'image-grid', 'image-grid']
                    metric_names = ['Loss'] \
                                   + ['PCA Video Projection {}'.format(ii)
                                      for ii in range(self.n_labels)] \
                                   + ['LDA Video Projection {}'.format(ii)
                                      for ii in range(self.n_labels)] \
                                   + ['t-SNE Video Projection {}'.format(ii)
                                      for ii in range(self.n_labels)] \
                                   + ['Video 1',
                                      'Video Reconstruction AE 1',
                                      'Video 2',
                                      'Video Reconstruction AE 2']

                else:
                    metrics_list = [self.plt_loss,
                                    self.tr_net_acc,
                                    self.val_net_acc] \
                                   + pca_projs \
                                   + lda_projs \
                                   + tsne_projs
                    iterations_list = [list(range(1, (self.current_epoch
                                                      * self.num_batches)+1)),
                                       list(range(1, self.current_epoch+1)),
                                       list(range(1, self.current_epoch+1))] \
                        + self.n_labels * [None] \
                        + self.n_labels * [None] \
                        + self.n_labels * [None]
                    plt_types = ['lines', 'lines', 'lines'] \
                        + self.n_labels * ['scatter'] \
                        + self.n_labels * ['scatter'] \
                        + self.n_labels * ['scatter']
                    metric_names = ['Loss',
                                    'Training Network Accuracy',
                                    'Validation Network Accuracy'] \
                        + ['PCA Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['LDA Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)] \
                        + ['t-SNE Video Projection {}'.format(ii)
                           for ii in range(self.n_labels)]

                plot_metrics(
                    metrics_list=metrics_list,
                    iterations_list=iterations_list,
                    types=plt_types,
                    metric_names=metric_names,
                    legend=True,
                    savefile=os.path.join(self.plt_dir,
                                          'metrics_epoch{}.png'
                                          .format(epoch)),
                    bot=self.bot)

                if self.plot_individually:
                    plot_metrics_individually(
                        metrics_list=metrics_list,
                        iterations_list=iterations_list,
                        types=plt_types,
                        metric_names=metric_names,
                        legend=True,
                        savefile=os.path.join(self.plt_dir,
                                              'metrics_epoch{}.png'
                                              .format(epoch)),
                        bot=None)

                if self.svm_analysis:
                    plot_metrics(
                        metrics_list=plt_acc,
                        iterations_list=metrics_it_list,
                        types=metrics_plt_types,
                        metric_names=acc_plt_names,
                        legend=True,
                        savefile=os.path.join(self.plt_dir,
                                              'acc_epoch{}.png'
                                              .format(epoch)),
                        bot=self.bot)
                    if self.plot_individually:
                        plot_metrics_individually(
                            metrics_list=plt_acc,
                            iterations_list=metrics_it_list,
                            types=metrics_plt_types,
                            metric_names=acc_plt_names,
                            legend=True,
                            savefile=os.path.join(self.plt_dir,
                                                  'acc_epoch{}.png'
                                                  .format(epoch)),
                            bot=None)

                    # plot_metrics(
                    #     metrics_list=plt_auc,
                    #     iterations_list=metrics_it_list,
                    #     types=metrics_plt_types,
                    #     metric_names=auc_plt_names,
                    #     legend=True,
                    #     # x_label='Iteration',
                    #     # y_label='Loss',
                    #     savefile=os.path.join(self.plt_dir,
                    #                           'auc_epoch{}.png'
                    #                           .format(epoch)),
                    #     bot=self.bot)
                    #
                    # if self.plot_individually:
                    #     plot_metrics_individually(
                    #         metrics_list=plt_auc,
                    #         iterations_list=metrics_it_list,
                    #         types=metrics_plt_types,
                    #         metric_names=auc_plt_names,
                    #         legend=True,
                    #         # x_label='Iteration',
                    #         # y_label='Loss',
                    #         savefile=os.path.join(self.plt_dir,
                    #                               'auc_epoch{}.png'
                    #                               .format(epoch)),
                    #         bot=None)

                confusion_matrices = []
                confusion_matrices_names = []
                for ii in range(self.n_classes):
                    confusion_matrices.append(
                        [confusion_matrix(val_labels[:, ii], val_svm_pred[:, ii]),
                         [-1, ii]])
                    confusion_matrices_names.append(
                        'Confusion Matrix - Class {}'.format(ii))

                plot_metrics(
                    metrics_list=confusion_matrices,
                    iterations_list=[None] * self.n_classes,
                    types=['confusion-matrix'] * self.n_classes,
                    metric_names=confusion_matrices_names,
                    legend=True,
                    savefile=os.path.join(self.plt_dir,
                                          'cm_epoch{}.png'
                                          .format(epoch)),
                    bot=self.bot)

                if self.plot_individually:
                    plot_metrics_individually(
                        metrics_list=confusion_matrices,
                        iterations_list=[None] * self.n_classes,
                        types=['confusion-matrix'] * self.n_classes,
                        metric_names=confusion_matrices_names,
                        legend=True,
                        savefile=os.path.join(self.plt_dir,
                                              'cm_epoch{}.png'
                                              .format(epoch)),
                        bot=None)

            ##################################################################
            # Checkpoints
            ##################################################################

            if epoch % self.checkpoint_epoch == 0 or \
                    epoch == self.epoch:
                n_epochs = self.checkpoint_epoch

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.plt_loss[-ii*self.num_batches:
                                      -(ii-1)*self.num_batches]
                        .reshape((1, -1)), index=[epoch-ii+1]) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'loss',
                                append=True, format='table')

                pd.DataFrame(
                    self.plt_loss[-self.num_batches:]
                    .reshape((1, -1)), index=[epoch]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'loss',
                            append=True, format='table')

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.val_loss[-ii*self.val_num_batches:
                                      -(ii-1)*self.val_num_batches]
                        .reshape((1, -1)), index=[epoch-ii+1]) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'val_loss',
                                append=True, format='table')

                pd.DataFrame(
                    self.val_loss[-self.val_num_batches:]
                    .reshape((1, -1)), index=[epoch]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_loss',
                            append=True, format='table')

                if self.is_ae is False:
                    pd.DataFrame(
                        self.tr_net_acc[-self.checkpoint_epoch:],
                        index=list(range(max(1, epoch-self.checkpoint_epoch+1),
                                         epoch+1))) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'tr_net_acc',
                                append=True, format='table')

                    pd.DataFrame(
                        self.val_net_acc[-self.checkpoint_epoch:],
                        index=list(range(max(1, epoch-self.checkpoint_epoch+1),
                                         epoch+1))) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'val_net_acc',
                                append=True, format='table')

                if self.svm_analysis:
                    pd.DataFrame(
                        self.val_acc[-1].reshape((1, -1)), index=[epoch]) \
                        .to_hdf(
                            os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_acc',
                            append=True, format='table')

                    # pd.DataFrame(
                    #     self.val_auc[-1].reshape((1, -1)), index=[epoch]) \
                    #     .to_hdf(
                    #         os.path.join(self.model_dir,
                    #                      'plt_loss_bkup.h5'),
                    #         'val_auc',
                    #         append=True, format='table')

                self.saver.save(
                    self.sess,
                    os.path.join(
                        self.model_dir,
                        'checkpoint-epoch'),
                    global_step=epoch)

            self.training_generator.on_epoch_end()
            self.current_epoch += 1

    def extract_features(self):
        if self.current_epoch == 1:
            print('Model needs to be trained before feature '
                  'extraction is done.')
        else:
            tr_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)

            for batch_number in tqdm(range(1, self.num_batches+1),
                                     position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm:
                    video_emb = \
                        self.sess.run(
                            self.test_video_emb,
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})
                else:
                    video_emb = \
                        self.sess.run(
                            self.video_emb,
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})

                tr_video_emb = np.vstack([tr_video_emb, video_emb])

            # Validation
            val_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)

            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.validation_generator.get_batch()

                if self.is_multilabel:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm:
                    video_emb = \
                        self.sess.run(
                            self.test_video_emb,
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})
                else:
                    video_emb = \
                        self.sess.run(
                            self.video_emb,
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})

                val_video_emb = np.vstack([val_video_emb,
                                           video_emb])

            pd.DataFrame(tr_video_emb) \
                .to_hdf(os.path.join(self.model_dir,
                                     'extracted_features.h5'),
                        'training_data')

            pd.DataFrame(val_video_emb) \
                .to_hdf(os.path.join(self.model_dir,
                                     'extracted_features.h5'),
                        'validation_data')

    def train(self):
        if self.current_epoch == 1:
            print('Model needs to be trained before evaluation is '
                  'performed.')
        else:
            if self.dataset_name in ['bouncingMNIST']:
                current_set = ['digits_only_slow', 'digits_only',
                               'velocity_only_slow', 'velocity_only',
                               'digits_and_velocity_slow', 'digits_and_velocity']
                generators = [(self.tr_digits_only_slow, self.val_digits_only_slow),
                              (self.tr_digits_only, self.val_digits_only),
                              (self.tr_velocity_only_slow, self.val_velocity_only_slow),
                              (self.tr_velocity_only, self.val_velocity_only),
                              (self.tr_digits_velocity_slow,
                               self.val_digits_velocity_slow),
                              (self.tr_digits_velocity, self.val_digits_velocity)]
            else:
                current_set = ['KTH', 'UCF101', 'YUP']
                generators = [(self.tr_KTH, self.val_KTH),
                              (self.tr_UCF101, self.val_UCF101),
                              (self.tr_YUP, self.val_YUP)]

            check_folder(
                os.path.join(self.plt_dir,
                             '{}_crossds_eval'.format(self.model_load_type)))

            for jj, (tr_generator, val_generator) in \
                    enumerate(generators):
                num_batches = tr_generator.dataset_size_ // \
                    tr_generator.batch_size_
                svm = SVMEval(tr_size=tr_generator.dataset_size_, val_size=val_generator.dataset_size_,
                              n_splits=5, scale=False, per_class=True,
                              n_labels=tr_generator.n_labels, verbose=self.verbosity)

                if self.is_cnn_lstm:
                    tr_cnn_video_emb = np.array([], dtype=np.float32) \
                        .reshape(0, self.cnn_emb_dim)
                    tr_cnn_labels = np.array([], dtype=np.float32) \
                        .reshape(0, tr_generator.n_classes)
                    tr_cnn_pred = np.array([], dtype=np.float32) \
                        .reshape(0, self.n_classes)

                tr_video_emb = np.array([], dtype=np.float32) \
                    .reshape(0, self.emb_dim)
                tr_labels = np.array([], dtype=np.float32) \
                    .reshape(0, tr_generator.n_classes)
                tr_pred = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)

                for batch_number in tqdm(range(num_batches), position=0):
                    video_batch, orig_labels = tr_generator.get_batch()

                    if self.dataset_name in ['bouncingMNIST']:
                        if self.is_cnn_lstm:
                            cnn_labels = orig_labels

                        inverted_labels = 1 - orig_labels
                        labels = orig_labels
                        inverted_labels = inverted_labels

                        labels = np.stack([inverted_labels, labels], axis=2)
                    else:
                        labels = np.zeros((tr_generator.batch_size_, self.n_classes), dtype=np.float32)

                    if self.is_ae:
                        if self.use_batch_norm:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss = \
                                        self.sess.run([self.test_cnn_loss],
                                                      feed_dict={self.video: video_batch,
                                                                 self.labels: labels})
                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.test_cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels = \
                                self.sess.run(
                                    [self.test_loss,
                                     self.test_video_emb,
                                     self.get_onehot_labels],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})
                        else:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss = \
                                        self.sess.run([self.cnn_loss],
                                                      feed_dict={self.video: video_batch,
                                                                 self.labels: labels})
                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels = \
                                self.sess.run(
                                    [self.loss,
                                     self.video_emb,
                                     self.get_onehot_labels],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})
                    else:
                        if self.use_batch_norm:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss, cnn_pred = \
                                        self.sess.run([self.test_cnn_loss,
                                                       self.get_onehot_test_cnn_pred],
                                                      feed_dict={self.video: video_batch,
                                                                 self.labels: labels})
                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.test_cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels, pred = \
                                self.sess.run(
                                    [self.test_loss,
                                     self.test_video_emb,
                                     self.get_onehot_labels,
                                     self.get_onehot_test_pred],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})
                        else:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss, cnn_pred = \
                                        self.sess.run([self.cnn_loss,
                                                       self.get_onehot_cnn_pred],
                                                      feed_dict={self.video: video_batch,
                                                                 self.labels: labels})
                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels, pred = \
                                self.sess.run(
                                    [self.loss,
                                     self.video_emb,
                                     self.get_onehot_labels,
                                     self.get_onehot_pred],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})

                    if self.is_cnn_lstm:
                        tr_cnn_video_emb = np.vstack([tr_cnn_video_emb, cnn_video_emb[15::16]])

                        if self.dataset_name in ['bouncingMNIST']:
                            tr_cnn_labels = np.vstack([tr_cnn_labels, cnn_labels])
                        else:
                            tr_cnn_labels = np.vstack([tr_cnn_labels, orig_labels])

                    tr_video_emb = np.vstack([tr_video_emb, video_emb])

                    if self.dataset_name in ['bouncingMNIST']:
                        tr_labels = np.vstack([tr_labels, labels])
                    else:
                        tr_labels = np.vstack([tr_labels, orig_labels])

                    if self.is_ae is False:
                        if self.is_cnn_lstm and not self.pretrained_cnn:
                            tr_cnn_pred = np.vstack([tr_cnn_pred,  cnn_pred])

                        tr_pred = np.vstack([tr_pred, pred])

                if self.is_ae is False and self.dataset_name in ['bouncingMNIST']:
                    if self.is_cnn_lstm and not self.pretrained_cnn:
                        tr_cnn_net_acc = accuracy_score(tr_cnn_labels, tr_cnn_pred)

                    tr_net_acc = accuracy_score(tr_labels, tr_pred)

                # Validation set

                num_batches = val_generator.dataset_size_ // \
                    val_generator.batch_size_

                if self.is_cnn_lstm:
                    val_cnn_video_emb = np.array([], dtype=np.float32) \
                        .reshape(0, self. cnn_emb_dim)
                    val_cnn_labels = np.array([], dtype=np.float32) \
                        .reshape(0, tr_generator.n_classes)
                    val_cnn_pred = np.array([], dtype=np.float32) \
                        .reshape(0, self.n_classes)

                val_video_emb = np.array([], dtype=np.float32) \
                    .reshape(0, self.emb_dim)
                val_labels = np.array([], dtype=np.float32) \
                    .reshape(0, tr_generator.n_classes)
                val_pred = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)

                for batch_number in tqdm(range(num_batches), position=0):
                    video_batch, orig_labels = val_generator.get_batch()

                    if self.dataset_name in ['bouncingMNIST']:
                        if self.is_cnn_lstm:
                            cnn_labels = orig_labels

                        inverted_labels = 1 - orig_labels
                        labels = orig_labels
                        inverted_labels = inverted_labels

                        labels = np.stack([inverted_labels, labels], axis=2)
                    else:
                        labels = np.zeros((tr_generator.batch_size_, self.n_classes), dtype=np.float32)

                    if self.is_ae:
                        if self.use_batch_norm:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss = self.sess.run([self.test_cnn_loss],
                                                             feed_dict={self.video: video_batch,
                                                                        self.labels: labels})

                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.test_cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels = \
                                self.sess.run(
                                    [self.test_loss,
                                     self.test_video_emb,
                                     self.get_onehot_labels],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})
                        else:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss = self.sess.run([self.cnn_loss],
                                                             feed_dict={self.video: video_batch,
                                                                        self.labels: labels})

                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels = \
                                self.sess.run(
                                    [self.loss,
                                     self.video_emb,
                                     self.get_onehot_labels],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})
                    else:
                        if self.use_batch_norm:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss, cnn_pred = \
                                        self.sess.run([self.test_cnn_loss,
                                                       self.get_onehot_test_cnn_pred],
                                                      feed_dict={self.video: video_batch,
                                                                 self.labels: labels})
                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.test_cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels, pred = \
                                self.sess.run(
                                    [self.test_loss,
                                     self.test_video_emb,
                                     self.get_onehot_labels,
                                     self.get_onehot_test_pred],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})
                        else:
                            if self.is_cnn_lstm:
                                if not self.pretrained_cnn:
                                    cnn_loss, cnn_pred = \
                                        self.sess.run([self.cnn_loss,
                                                       self.get_onehot_cnn_pred],
                                                      feed_dict={self.video: video_batch,
                                                                 self.labels: labels})
                                cnn_video_emb, cnn_labels = \
                                    self.sess.run(
                                        [self.cnn_feats,
                                         self.get_onehot_labels],
                                        feed_dict={self.video: video_batch,
                                                   self.labels: labels})

                            loss, video_emb, labels, pred = \
                                self.sess.run(
                                    [self.loss,
                                     self.video_emb,
                                     self.get_onehot_labels,
                                     self.get_onehot_pred],
                                    feed_dict={self.video: video_batch,
                                               self.labels: labels})

                    if self.is_cnn_lstm:
                        val_cnn_video_emb = np.vstack([val_cnn_video_emb, cnn_video_emb[15::16]])

                        if self.dataset_name in ['bouncingMNIST']:
                            val_cnn_labels = np.vstack([val_cnn_labels, cnn_labels])
                        else:
                            val_cnn_labels = np.vstack([val_cnn_labels, orig_labels])

                    val_video_emb = np.vstack([val_video_emb, video_emb])

                    if self.dataset_name in ['bouncingMNIST']:
                        val_labels = np.vstack([val_labels, labels])
                    else:
                        val_labels = np.vstack([val_labels, orig_labels])

                    if self.is_ae is False:
                        if self.is_cnn_lstm and not self.pretrained_cnn:
                            val_cnn_pred = np.vstack([val_cnn_pred, cnn_pred])
                        val_pred = np.vstack([val_pred, pred])

                if self.is_ae is False and self.dataset_name in ['bouncingMNIST']:
                    if self.is_cnn_lstm and not self.pretrained_cnn:
                        val_cnn_net_acc = accuracy_score(val_cnn_labels, val_cnn_pred)
                    val_net_acc = accuracy_score(val_labels, val_pred)

                if self.svm_analysis:
                    if self.is_cnn_lstm:
                        val_cnn_acc, val_svm_cnn_pred, valid_cnn_classes = \
                            svm.compute(train_data=(tr_cnn_video_emb, tr_cnn_labels),
                                        val_data=(val_cnn_video_emb, val_cnn_labels))
                    # val_auc,
                    val_acc, val_svm_pred, valid_classes = \
                        svm.compute(train_data=(tr_video_emb, tr_labels),
                                    val_data=(val_video_emb, val_labels))

                if self.is_cnn_lstm:
                    val_labels_n = get_labels(val_cnn_labels, n_labels=self.n_labels)
                    pca_projs = []
                    lda_projs = []
                    tsne_projs = []
                    for ii in range(self.n_labels):
                        label = val_labels_n[:, ii].reshape((-1, 1))

                        # PCA
                        if ii == 0:
                            pca = PCA(n_components=2, whiten=True)
                            pca_video_proj = pca.fit_transform(val_cnn_video_emb)

                        pca_video_proj_aux = np.concatenate([pca_video_proj,
                                                             label],
                                                            axis=1)
                        pca_projs.append(pca_video_proj_aux)

                        # LDA
                        lda = LinearDiscriminantAnalysis(n_components=2)
                        lda_video_proj = lda.fit_transform(val_cnn_video_emb,
                                                           label.squeeze())
                        lda_video_proj = np.concatenate([lda_video_proj, label],
                                                        axis=1)
                        lda_projs.append(lda_video_proj)

                        # t-SNE
                        if ii == 0:
                            tsne = TSNE(n_components=2)
                            tsne_video_proj = tsne.fit_transform(val_cnn_video_emb)

                            tsne_video_proj = \
                                tsne_video_proj[
                                    :self.validation_generator.dataset_size_]

                        tsne_video_proj_aux = np.concatenate([tsne_video_proj,
                                                              label],
                                                             axis=1)
                        tsne_projs.append(tsne_video_proj_aux)

                    if self.is_ae:
                        metrics_list = pca_projs \
                                       + lda_projs \
                                       + tsne_projs \
                                       + [video_batch[0].squeeze(),
                                          net_out[0].squeeze(),
                                          video_batch[1].squeeze(),
                                          net_out[1].squeeze()]
                        iterations_list = self.n_labels * [None] \
                                          + self.n_labels * [None] \
                                          + self.n_labels * [None] \
                                          + [None, None, None, None]
                        plt_types = self.n_labels * ['scatter'] \
                                    + self.n_labels * ['scatter'] \
                                    + self.n_labels * ['scatter'] \
                                    + ['image-grid', 'image-grid',
                                       'image-grid', 'image-grid']
                        metric_names = ['PCA Video Projection {}'.format(ii)
                                        for ii in range(1, self.n_labels+1)] \
                                       + ['LDA Video Projection {}'.format(ii)
                                          for ii in range(1, self.n_labels+1)] \
                                       + ['t-SNE Video Projection {}'.format(ii)
                                          for ii in range(1, self.n_labels+1)] \
                                       + ['Video 1',
                                          'Video Reconstruction AE 1',
                                          'Video 2',
                                          'Video Reconstruction AE 2']

                    else:
                        if not self.pretrained_cnn and self.dataset_name in ['bouncingMNIST']:
                            metrics_list = [[tr_cnn_net_acc,
                                            val_cnn_net_acc]]
                            iterations_list = [['Training Accuracy',
                                               'Validation Accuracy']]
                            plt_types = ['lines']
                            metric_names = ['Network Accuracy']
                        else:
                            metrics_list = []
                            iterations_list = []
                            plt_types = []
                            metric_names = []

                        metrics_list = metrics_list \
                                       + pca_projs \
                                       + lda_projs \
                                       + tsne_projs
                        iterations_list = iterations_list \
                                          + self.n_labels * [None] \
                                          + self.n_labels * [None] \
                                          + self.n_labels * [None]
                        plt_types = plt_types \
                                    + self.n_labels * ['scatter'] \
                                    + self.n_labels * ['scatter'] \
                                    + self.n_labels * ['scatter']
                        metric_names = metric_names \
                                       + ['PCA Video Projection {}'.format(ii)
                                          for ii in range(1, self.n_labels+1)] \
                                       + ['LDA Video Projection {}'.format(ii)
                                          for ii in range(1, self.n_labels+1)] \
                                       + ['t-SNE Video Projection {}'.format(ii)
                                          for ii in range(1, self.n_labels+1)]

                    if not self.pretrained_cnn and self.dataset_name in ['bouncingMNIST']:
                        if jj == 0:
                            pd.DataFrame(
                                [{'dataset': current_set[jj],
                                  'training_set_acc': tr_cnn_net_acc,
                                  'validation_set_acc': val_cnn_net_acc}], index=[jj]) \
                                .to_hdf(os.path.join(self.model_dir,
                                                     'plt_loss_bkup.h5'),
                                        'crossds_cnn_net_acc', format='table',
                                        min_itemsize={'dataset': 100})
                        else:
                            pd.DataFrame(
                                [{'dataset': current_set[jj],
                                  'training_set_acc': tr_cnn_net_acc,
                                  'validation_set_acc': val_cnn_net_acc}], index=[jj]) \
                                .to_hdf(os.path.join(self.model_dir,
                                                     'plt_loss_bkup.h5'),
                                        'crossds_cnn_net_acc',
                                        append=True, format='table',
                                        min_itemsize={'dataset': 100})

                    plot_metrics(
                        metrics_list=metrics_list,
                        iterations_list=iterations_list,
                        types=plt_types,
                        metric_names=metric_names,
                        legend=True,
                        savefile=os.path.join(self.plt_dir,
                                              '{}_crossds_eval'
                                              .format(self.model_load_type),
                                              'crossds_{}_cnn_metrics.png'
                                              .format(
                                                current_set[jj].replace(' ',
                                                                        '-'))),
                        bot=self.bot)

                    if self.plot_individually:
                        plot_metrics_individually(
                            metrics_list=metrics_list,
                            iterations_list=iterations_list,
                            types=plt_types,
                            metric_names=metric_names,
                            legend=True,
                            savefile=
                                os.path.join(self.plt_dir,
                                             '{}_crossds_eval'
                                             .format(self.model_load_type),
                                            'crossds_{}_cnn_metrics.png'
                                             .format(
                                                current_set[jj].replace(' ',
                                                                        '-'))),
                            bot=None)

                    if self.svm_analysis:
                        plot_metrics(
                            metrics_list=[val_cnn_acc],
                            iterations_list=[list(range(1, tr_generator.n_classes+1))],
                            types=['lines'],
                            metric_names=['Per Class Validation Accuracy'
                                          ' (SVM)'],
                            legend=True,
                            savefile=
                                os.path.join(self.plt_dir,
                                             '{}_crossds_eval'
                                             .format(self.model_load_type),
                                             'crossds_{}_cnn_acc.png'
                                              .format(
                                                current_set[jj].replace(' ',
                                                                        '-'))),
                            bot=self.bot)

                    confusion_matrices = []
                    confusion_matrices_names = []
                    for ii in range(tr_generator.n_classes):
                        confusion_matrices.append(
                            [confusion_matrix(val_cnn_labels[:, ii], val_svm_cnn_pred[:, ii]), #  val_cnn_pred[:, ii]),
                             [-1, ii]])
                        confusion_matrices_names.append(
                            'Confusion Matrix - Class {}'.format(ii))

                    pd.DataFrame(
                        [{'dataset': current_set[jj],
                          'val_labels': val_cnn_labels,
                          'val_pred': val_svm_cnn_pred,
                          'conf_matrices': confusion_matrices_names}], index=[jj])\
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'crossds_pred_cnn_labels_{}'
                                    .format(current_set[jj]),
                                min_itemsize={'dataset': 100})

                    plot_metrics(
                        metrics_list=confusion_matrices,
                        iterations_list=[None] * tr_generator.n_classes,
                        types=['confusion-matrix'] * tr_generator.n_classes,
                        metric_names=confusion_matrices_names,
                        legend=True,
                        savefile=os.path.join(self.plt_dir,
                                              '{}_crossds_eval'
                                              .format(self.model_load_type),
                                              'crossds_{}_cnn_cm.png'
                                              .format(
                                                current_set[jj].replace(' ',
                                                                        '-'))),
                        bot=self.bot)

                    if self.plot_individually:
                        plot_metrics_individually(
                            metrics_list=confusion_matrices,
                            iterations_list=[None] * tr_generator.n_classes,
                            types=['confusion-matrix'] * tr_generator.n_classes,
                            metric_names=confusion_matrices_names,
                            legend=True,
                            savefile=
                                os.path.join(self.plt_dir,
                                             '{}_crossds_eval'
                                             .format(self.model_load_type),
                                             'crossds_{}_cnn_cm.png'
                                             .format(
                                                current_set[jj].replace(' ',
                                                                        '-'))),
                            bot=None)

                val_labels_n = get_labels(val_labels, n_labels=self.n_labels)
                pca_projs = []
                lda_projs = []
                tsne_projs = []
                for ii in range(self.n_labels):
                    label = val_labels_n[:, ii].reshape((-1, 1))

                    # PCA
                    if ii == 0:
                        pca = PCA(n_components=2, whiten=True)
                        pca_video_proj = pca.fit_transform(val_video_emb)

                    pca_video_proj_aux = np.concatenate([pca_video_proj,
                                                         label],
                                                        axis=1)
                    pca_projs.append(pca_video_proj_aux)

                    # LDA
                    try:
                        lda = LinearDiscriminantAnalysis(n_components=2)
                        lda_video_proj = lda.fit_transform(val_video_emb,
                                                           label.squeeze())
                        lda_video_proj = np.concatenate([lda_video_proj, label],
                                                        axis=1)
                        lda_projs.append(lda_video_proj)
                    except np.linalg.LinAlgError:
                        pass

                    # t-SNE
                    if ii == 0:
                        tsne = TSNE(n_components=2)
                        tsne_video_proj = tsne.fit_transform(val_video_emb)

                        tsne_video_proj = \
                            tsne_video_proj[
                                :self.validation_generator.dataset_size_]

                    tsne_video_proj_aux = np.concatenate([tsne_video_proj,
                                                          label],
                                                         axis=1)
                    tsne_projs.append(tsne_video_proj_aux)

                if self.is_ae:
                    metrics_list = pca_projs \
                                   + lda_projs \
                                   + tsne_projs \
                                   + [video_batch[0].squeeze(),
                                      net_out[0].squeeze(),
                                      video_batch[1].squeeze(),
                                      net_out[1].squeeze()]
                    iterations_list = self.n_labels * [None] \
                                      + self.n_labels * [None] \
                                      + self.n_labels * [None] \
                                      + [None, None, None, None]
                    plt_types = self.n_labels * ['scatter'] \
                                + self.n_labels * ['scatter'] \
                                + self.n_labels * ['scatter'] \
                                + ['image-grid', 'image-grid',
                                   'image-grid', 'image-grid']
                    metric_names = ['PCA Video Projection {}'.format(ii)
                                    for ii in range(1, self.n_labels+1)] \
                                   + ['LDA Video Projection {}'.format(ii)
                                      for ii in range(1, self.n_labels+1)] \
                                   + ['t-SNE Video Projection {}'.format(ii)
                                      for ii in range(1, self.n_labels+1)] \
                                   + ['Video 1',
                                      'Video Reconstruction AE 1',
                                      'Video 2',
                                      'Video Reconstruction AE 2']

                else:
                    if self.dataset_name in ['bouncingMNIST']:
                        metrics_list = [[tr_net_acc,
                                        val_net_acc]]
                        iterations_list = [['Training Accuracy',
                                           'Validation Accuracy']]
                        plt_types = ['lines']
                        metric_names = ['Network Accuracy']
                    else:
                        metrics_list = []
                        iterations_list = []
                        plt_types = []
                        metric_names = []

                    metrics_list = metrics_list \
                                   + pca_projs \
                                   + lda_projs \
                                   + tsne_projs
                    iterations_list = iterations_list \
                                      + self.n_labels * [None] \
                                      + self.n_labels * [None] \
                                      + self.n_labels * [None]
                    plt_types = plt_types \
                                + self.n_labels * ['scatter'] \
                                + self.n_labels * ['scatter'] \
                                + self.n_labels * ['scatter']
                    metric_names = metric_names \
                                   + ['PCA Video Projection {}'.format(ii)
                                      for ii in range(1, self.n_labels+1)] \
                                   + ['LDA Video Projection {}'.format(ii)
                                      for ii in range(1, self.n_labels+1)] \
                                   + ['t-SNE Video Projection {}'.format(ii)
                                      for ii in range(1, self.n_labels+1)]

                if self.dataset_name in ['bouncingMNIST']:
                    if jj == 0:
                        pd.DataFrame(
                            [{'dataset': current_set[jj],
                              'training_set_acc': tr_net_acc,
                              'validation_set_acc': val_net_acc}], index=[jj]) \
                            .to_hdf(os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'crossds_net_acc', format='table',
                                    min_itemsize={'dataset': 100})
                    else:
                        pd.DataFrame(
                            [{'dataset': current_set[jj],
                              'training_set_acc': tr_net_acc,
                              'validation_set_acc': val_net_acc}], index=[jj]) \
                            .to_hdf(os.path.join(self.model_dir,
                                                 'plt_loss_bkup.h5'),
                                    'crossds_net_acc',
                                    append=True, format='table',
                                    min_itemsize={'dataset': 100})

                plot_metrics(
                    metrics_list=metrics_list,
                    iterations_list=iterations_list,
                    types=plt_types,
                    metric_names=metric_names,
                    legend=True,
                    savefile=os.path.join(self.plt_dir,
                                          '{}_crossds_eval'
                                          .format(self.model_load_type),
                                          'crossds_{}_metrics.png'
                                          .format(
                                            current_set[jj].replace(' ',
                                                                    '-'))),
                    bot=self.bot)

                if self.plot_individually:
                    plot_metrics_individually(
                        metrics_list=metrics_list,
                        iterations_list=iterations_list,
                        types=plt_types,
                        metric_names=metric_names,
                        legend=True,
                        savefile=
                            os.path.join(self.plt_dir,
                                         '{}_crossds_eval'
                                         .format(self.model_load_type),
                                        'crossds_{}_metrics.png'
                                         .format(
                                            current_set[jj].replace(' ',
                                                                    '-'))),
                        bot=None)

                if self.svm_analysis:
                    plot_metrics(
                        metrics_list=[val_acc],
                        iterations_list=[list(range(1, tr_generator.n_classes+1))],
                        types=['lines'],
                        metric_names=['Per Class Validation Accuracy'
                                      ' (SVM)'],
                        legend=True,
                        savefile=
                            os.path.join(self.plt_dir,
                                         '{}_crossds_eval'
                                         .format(self.model_load_type),
                                         'crossds_{}_acc.png'
                                          .format(
                                            current_set[jj].replace(' ',
                                                                    '-'))),
                        bot=self.bot)

                confusion_matrices = []
                confusion_matrices_names = []
                for ii in range(tr_generator.n_classes):
                    confusion_matrices.append(
                        [confusion_matrix(val_labels[:, ii], val_svm_pred[:, ii]),
                         [-1, ii]])
                    confusion_matrices_names.append(
                        'Confusion Matrix - Class {}'.format(ii))

                pd.DataFrame(
                    [{'dataset': current_set[jj],
                      'val_labels': val_labels,
                      'val_pred': val_svm_pred,
                      'conf_matrices': confusion_matrices_names}], index=[jj])\
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'crossds_pred_labels_{}'
                                .format(current_set[jj]),
                            min_itemsize={'dataset': 100})

                plot_metrics(
                    metrics_list=confusion_matrices,
                    iterations_list=[None] * tr_generator.n_classes,
                    types=['confusion-matrix'] * tr_generator.n_classes,
                    metric_names=confusion_matrices_names,
                    legend=True,
                    savefile=os.path.join(self.plt_dir,
                                          '{}_crossds_eval'
                                          .format(self.model_load_type),
                                          'crossds_{}_cm.png'
                                          .format(
                                            current_set[jj].replace(' ',
                                                                    '-'))),
                    bot=self.bot)

                if self.plot_individually:
                    plot_metrics_individually(
                        metrics_list=confusion_matrices,
                        iterations_list=[None] * tr_generator.n_classes,
                        types=['confusion-matrix'] * tr_generator.n_classes,
                        metric_names=confusion_matrices_names,
                        legend=True,
                        savefile=
                            os.path.join(self.plt_dir,
                                         '{}_crossds_eval'
                                         .format(self.model_load_type),
                                         'crossds_{}_cm.png'
                                         .format(
                                            current_set[jj].replace(' ',
                                                                    '-'))),
                        bot=None)


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu:
        gpu_options = tf.GPUOptions(visible_device_list=args.gpu_id,
                                    allow_growth=True)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                      inter_op_parallelism_threads=1,
                                      gpu_options=gpu_options,
                                      allow_soft_placement=True)
    else:
        session_conf = tf.ConfigProto()

    sess = tf.Session(config=session_conf)

    model = VideoRep(sess=sess,
                     model_id=args.model_id,
                     model_name=args.model_name,
                     model_type=args.model_type,
                     ex_mode=args.ex_mode,
                     model_load_type=args.model_load_type,
                     dataset_name=args.dataset_name,
                     class_velocity=args.class_velocity,
                     hide_digits=args.hide_digits,
                     step_length=args.step_length,
                     tr_size=args.tr_size,
                     val_size=args.val_size,
                     use_batch_norm=args.use_batch_norm,
                     use_layer_norm=args.use_layer_norm,
                     use_l2_reg=args.use_l2_reg,
                     pretrained_cnn=args.pretrained_cnn,
                     epoch=args.epoch,
                     batch_size=args.batch_size,
                     learning_rate=args.learning_rate,
                     svm_analysis=args.svm_analysis,
                     vis_epoch=args.vis_epoch,
                     plot_individually=args.plot_individually,
                     verbosity=args.verbosity,
                     checkpoint_epoch=args.checkpoint_epoch,
                     keep_checkpoint_max=args.keep_checkpoint_max,
                     redo=args.redo,
                     slack_bot=args.slack_bot,
                     plt_dir=args.plt_dir,
                     model_dir=args.model_dir,
                     config_dir=args.config_dir)
    if args.ex_mode == 'training':
        model.train_model()
    elif args.ex_mode == 'feat_extraction':
        model.extract_features()
    elif args.ex_mode == 'train':
        model.train()
    else:
        raise NotImplementedError
