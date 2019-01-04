import os
import sys
import time
import configparser
import numpy as np
import pandas as pd
import random as rn

import tensorflow as tf

from tqdm import tqdm

from parse_config import parse_args

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score

from models_c3d import def_c3d_large_video_ae, \
                       def_c3d_small_video_ae, \
                       def_c3d_large_video_classifier, \
                       def_c3d_small_video_classifier
from models_lstm import def_lstm_large_video_ae, def_lstm_small_video_ae
from models_gru import def_gru_large_video_ae, def_gru_small_video_ae
from models_residual import def_r3d

from bouncingMNIST import BouncingMNISTDataGenerator
from utils import plot_metrics, plot_metrics_individually, get_labels, SVMEval

tl = tf.layers

os.environ["PYTHONHASHSEED"] = '42'
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


class VideoRep():

    def __init__(self, sess,
                 model_id, model_name, model_type,
                 dataset_name, class_velocity, hide_digits,
                 tr_size, val_size,
                 use_batch_norm, use_layer_norm, use_l2_reg,
                 epoch, batch_size, learning_rate,
                 svm_analysis, vis_epoch, plot_individually, verbosity,
                 checkpoint_epoch, keep_checkpoint_max, redo,
                 slack_bot,
                 plt_dir, model_dir, config_dir):
        self.sess = sess

        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.is_ae = ('ae' in self.model_type)

        self.dataset_name = dataset_name
        if self.dataset_name in ['bouncingMNIST']:
            self.is_multilabel = True
        else:
            self.is_multilabel = False
        self.class_velocity = class_velocity
        self.hide_digits = hide_digits
        self.tr_size = tr_size
        self.val_size = val_size
        self.n_classes = 10                 # number of possible classes
        self.n_labels = 2                   # number of labels for each sample

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_l2_reg = use_l2_reg

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

        if slack_bot is True:
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

        if self.dataset_name == 'bouncingMNIST':
            self.training_generator = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.tr_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    noise=self.is_ae,
                    class_velocity=self.class_velocity,
                    hide_digits=self.hide_digits)
            self.validation_generator = \
                BouncingMNISTDataGenerator(
                    dataset_size=self.val_size,
                    batch_size=self.batch_size,
                    ae=self.is_ae,
                    split='test',
                    class_velocity=self.class_velocity,
                    hide_digits=self.hide_digits)

        self.num_batches = \
            self.training_generator.dataset_size_ // \
            self.training_generator.batch_size_
        self.val_num_batches = \
            self.validation_generator.dataset_size_ // \
            self.validation_generator.batch_size_

        self.saver = tf.train.Saver(max_to_keep=keep_checkpoint_max)

        if self.redo is not True:
            latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if latest_checkpoint is not None:
                if self.verbosity >= 1:
                    print('Loading checkpoint - {}'.format(latest_checkpoint))
                self.saver.restore(self.sess, latest_checkpoint)

                # Load training losses from previous checkpoint
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
                    self.val_net_acc = pd.read_hdf(
                        os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                        'val_net_acc').values \
                        .flatten()

                if self.svm_analysis is True:
                    self.val_auc = pd.read_hdf(
                        os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                        'val_auc').values

                    self.val_acc = pd.read_hdf(
                        os.path.join(self.model_dir, 'plt_loss_bkup.h5'),
                        'val_acc').values

                self.current_epoch = (self.plt_loss.shape[0] //
                                      self.num_batches)
                if self.verbosity >= 1:
                    print('Checkpoint loaded - epoch {}'
                          .format(self.current_epoch))

        if self.current_epoch == 1:
            self.plt_loss = np.array([])
            self.val_loss = np.array([])

            if self.is_ae is False:
                self.val_net_acc = np.array([])

            if self.svm_analysis is True:
                self.val_auc = np.array([]) \
                    .reshape(0, self.n_classes)
                self.val_acc = np.array([]) \
                    .reshape(0, self.n_classes)

    def _get_model_name(self):
        # return 'semantic_nn_{}_ae_{}_{}emb_{}epoch_{}' \
        #     .format(self.comb_type,
        #             self.ae_type,
        #             self.emb_dim,
        #             self.epoch,
        #             model_name)
        return '{}_{}'.format(self.model_name, self.model_id)

    def _def_loss_fn(self):
        if self.is_ae is True:  # mse
            self.loss = tf.reduce_mean(
                tf.square(self.net_out - self.video))

        else:  # binary_crossentropy
            if self.dataset_name in ['bouncingMNIST']:
                self.loss = tf.reduce_mean(tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.net_out_logits,
                        labels=self.labels), axis=1))
            else:
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.net_out_logits,
                        labels=self.labels))

        if self.use_batch_norm is True:
            if self.is_ae is True:  # mse
                self.test_loss = tf.reduce_mean(
                    tf.square(self.test_net_out - self.video))

            else:  # binary_crossentropy
                if self.dataset_name in ['bouncingMNIST']:
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
            optim = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                       beta1=self.beta1) \
                .minimize(self.loss,
                          var_list=self.learnable_vars)

            return optim

    def _def_svm_clf(self):
        self.svm = SVMEval(tr_size=self.tr_size, val_size=self.val_size,
                           n_splits=5, scale=False, per_class=True,
                           verbose=self.verbosity)

    def _def_eval_metrics(self):
        if self.is_ae is True:
            pass
        else:
            if self.is_multilabel is True:
                labels = tf.squeeze(tf.nn.top_k(self.labels).indices)
                predictions = tf.squeeze(tf.nn.top_k(self.net_out).indices)

                correct_predictions = tf.equal(labels, predictions)
                mean_acc = \
                    tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                overall_acc = \
                    tf.reduce_mean(tf.reduce_min(
                        tf.cast(correct_prediction, tf.float32), axis=1))

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
        """ INPUTS """
        self.video = tf.placeholder(tf.float32, [None, 16, 64, 64, 1])
        if self.is_multilabel is True:
            self.labels = tf.placeholder(tf.float32, [None, self.n_classes, 2])
        else:
            self.labels = tf.placeholder(tf.float32, [None, self.n_classes])

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
                def_c3d_small_video_classifier(
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
        elif self.model_type == 'p3d_ae_small':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_p3d_small_video_ae(
                    self.video,
                    is_training=True,
                    reuse=False,
                    use_l2_reg=self.use_l2_reg,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm)
        elif self.model_type == 'p3d_ae_large':
            self.net_out, \
                self.net_out_logits, \
                self.video_emb, \
                self.learnable_vars, \
                self.emb_dim = \
                def_p3d_large_video_ae(
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

        if self.use_batch_norm is True:
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
                    def_c3d_small_video_classifier(
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
            elif self.model_type == 'p3d_ae_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_p3d_small_video_ae(self.video,
                                           is_training=False,
                                           reuse=True)
            elif self.model_type == 'p3d_ae_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_p3d_large_video_ae(self.video,
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

        self._def_loss_fn()
        if self.svm_analysis is True:
            self._def_svm_clf()
        # if self.is_ae is False:
        #     self.init_eval_vars = self._def_eval_metrics()
        self.optim = self._def_optimizer()

    def train_model(self):
        if self.current_epoch == 1:
            self.sess.run(tf.global_variables_initializer())

        start_time = time.time()

        starting_epoch = self.current_epoch

        for epoch in tqdm(range(starting_epoch, self.epoch+1), position=1):
            # Training
            tr_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)

            if self.is_multilabel is True:
                tr_labels = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)
                tr_pred = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)
            else:
                tr_labels = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)
                tr_pred = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)

            for batch_number in tqdm(range(1, self.num_batches+1),
                                     position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                if self.is_multilabel is True:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                _, loss, net_out = \
                    self.sess.run([self.optim,
                                   self.loss,
                                   self.net_out],
                                  feed_dict={self.video: video_batch,
                                             self.labels: labels})

                if self.verbosity >= 1:
                    print('Epoch: [%2d] [%4d / %4d] time: %4.4f,'
                          ' loss: %.8f'
                          % (epoch, batch_number, self.num_batches,
                             time.time() - start_time,
                             loss))

                self.plt_loss = np.append(self.plt_loss, loss)

                if self.is_ae is True:
                    if self.use_batch_norm is True:
                        video_emb, labels = \
                            self.sess.run(
                                [self.test_video_emb,
                                 tf.squeeze(tf.nn.top_k(labels).indices)],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        video_emb, labels = \
                            self.sess.run(
                                [self.video_emb,
                                 tf.squeeze(tf.nn.top_k(labels).indices)],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                else:
                    if self.use_batch_norm is True:
                        video_emb, labels, pred = \
                            self.sess.run(
                                [self.test_video_emb,
                                 tf.squeeze(tf.nn.top_k(labels).indices),
                                 tf.squeeze(
                                    tf.nn.top_k(self.test_net_out).indices)],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        video_emb, labels, pred = \
                            self.sess.run(
                                [self.video_emb,
                                 tf.squeeze(tf.nn.top_k(labels).indices),
                                 tf.squeeze(
                                    tf.nn.top_k(self.net_out).indices)],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})

                tr_video_emb = np.vstack([tr_video_emb, video_emb])
                tr_labels = np.vstack([tr_labels, labels])
                if self.is_ae is False:
                    tr_pred = np.vstack([tr_pred, pred])

            # Validation and Visualization
            val_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)

            if self.is_multilabel is True:
                val_labels = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)
                val_pred = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)
            else:
                val_labels = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)
                val_pred = np.array([], dtype=np.float32) \
                    .reshape(0, self.n_classes)

            # if self.is_ae is False:
            #     self.sess.run([self.init_eval_vars])

            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.validation_generator.get_batch()

                if self.is_multilabel is True:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm is True:
                    loss, net_out, video_emb = \
                        self.sess.run(
                            [self.test_loss,
                             self.test_net_out,
                             self.test_video_emb],
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})
                else:
                    loss, net_out, video_emb = \
                        self.sess.run(
                            [self.loss,
                             self.net_out,
                             self.video_emb],
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})

                if self.is_ae is True:
                    labels = \
                        self.sess.run(
                            tf.squeeze(tf.nn.top_k(labels).indices),
                            feed_dict={self.video: video_batch,
                                       self.labels: labels})
                else:
                    if self.use_batch_norm is True:
                        labels, pred = \
                            self.sess.run(
                                [tf.squeeze(tf.nn.top_k(labels).indices),
                                 tf.squeeze(
                                    tf.nn.top_k(self.test_net_out).indices)],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})
                    else:
                        labels, pred = \
                            self.sess.run(
                                [tf.squeeze(tf.nn.top_k(labels).indices),
                                 tf.squeeze(
                                    tf.nn.top_k(self.net_out).indices)],
                                feed_dict={self.video: video_batch,
                                           self.labels: labels})

                self.val_loss = np.append(self.val_loss, loss)
                val_video_emb = np.vstack([val_video_emb, video_emb])
                val_labels = np.vstack([val_labels, labels])
                if self.is_ae is False:
                    val_pred = np.vstack([val_pred, pred])

            if self.is_ae is False:
                # TODO: Fix problem with one-hot input to classification_report
                # if self.verbosity >= 2:
                #     print("Validation classification report - Epoch {}:"
                #           .format(epoch))
                #     print(classification_report(val_labels, val_pred))
                self.val_net_acc = np.append(
                    self.val_net_acc,
                    accuracy_score(val_labels, val_pred))
            self.validation_generator.on_epoch_end()

            if epoch % self.vis_epoch == 0 or \
                    epoch == 1 or \
                    epoch == self.epoch:

                if self.svm_analysis is True:
                    val_acc, val_auc, valid_classes = \
                        self.svm.compute(train_data=(tr_video_emb, tr_labels),
                                         val_data=(val_video_emb, val_labels))
                    self.val_acc = np.vstack([self.val_acc, val_acc])
                    self.val_auc = np.vstack([self.val_auc, val_auc])

                    plt_acc = []
                    plt_auc = []
                    metrics_it_list = []
                    acc_plt_names = []
                    auc_plt_names = []
                    metrics_plt_types = []
                    for cl in range(self.n_classes):
                        plt_acc.append(self.val_acc[:, cl])
                        plt_auc.append(self.val_auc[:, cl])
                        if self.vis_epoch == 1:
                            metrics_it_list.append(
                                list(range(self.vis_epoch,
                                           self.current_epoch+1,
                                           self.vis_epoch)))
                        elif epoch != self.epoch:
                            metrics_it_list.append(
                                [1] +
                                list(range(self.vis_epoch,
                                           self.current_epoch+1,
                                           self.vis_epoch)))
                        elif epoch == self.epoch and \
                                epoch % self.vis_epoch != 0:
                            metrics_it_list.append(
                                [1] +
                                list(range(self.vis_epoch,
                                           self.current_epoch+1,
                                           self.vis_epoch)) +
                                [epoch])
                        else:
                            metrics_it_list.append(
                                [1] +
                                list(range(self.vis_epoch,
                                           self.current_epoch+1,
                                           self.vis_epoch)))
                        metrics_plt_types.append('lines')
                        acc_plt_names.append('Validation Accuracy (SVM) - '
                                             'Class {}'.format(cl))
                        auc_plt_names.append('Validation ROC AUC (SVM) - '
                                             'Class {}'.format(cl))

                val_labels_n = get_labels(val_labels, n_labels=self.n_labels)
                label1 = val_labels_n[:, 0].reshape((-1, 1))
                label2 = val_labels_n[:, 1].reshape((-1, 1))

                # PCA
                pca = PCA(n_components=2, whiten=True)
                pca_video_proj = pca.fit_transform(val_video_emb)
                pca_video_proj1 = np.concatenate([pca_video_proj, label1],
                                                 axis=1)
                pca_video_proj2 = np.concatenate([pca_video_proj, label2],
                                                 axis=1)
                # LDA
                lda = LinearDiscriminantAnalysis(n_components=2)
                lda_video_proj1 = lda.fit_transform(val_video_emb,
                                                    label1.squeeze())
                lda_video_proj1 = np.concatenate([lda_video_proj1, label1],
                                                 axis=1)

                lda_video_proj2 = lda.fit_transform(val_video_emb,
                                                    label2.squeeze())
                lda_video_proj2 = np.concatenate([lda_video_proj2, label2],
                                                 axis=1)
                # t-SNE
                tsne = TSNE(n_components=2)
                tsne_video_proj = tsne.fit_transform(val_video_emb)
                tsne_video_proj = \
                    tsne_video_proj[:self.validation_generator.dataset_size_]

                tsne_video_proj1 = np.concatenate([tsne_video_proj, label1],
                                                  axis=1)
                tsne_video_proj2 = np.concatenate([tsne_video_proj, label2],
                                                  axis=1)

                if self.is_ae is True:
                    metrics_list = [self.plt_loss,
                                    pca_video_proj1,
                                    pca_video_proj2,
                                    lda_video_proj1,
                                    lda_video_proj2,
                                    tsne_video_proj1,
                                    tsne_video_proj2,
                                    video_batch[0].squeeze(),
                                    net_out[0].squeeze(),
                                    video_batch[1].squeeze(),
                                    net_out[1].squeeze()]
                    iterations_list = [list(range(1, (self.current_epoch *
                                                  self.num_batches)+1)),
                                       None, None, None, None,
                                       None, None, None,
                                       None, None, None]
                    plt_types = ['lines',
                                 'scatter', 'scatter',
                                 'scatter', 'scatter', 'scatter',
                                 'scatter', 'image-grid', 'image-grid',
                                 'image-grid', 'image-grid']
                    metric_names = ['Loss',
                                    'PCA Video Projection 1',
                                    'PCA Video Projection 2',
                                    'LDA Video Projection 1',
                                    'LDA Video Projection 2',
                                    't-SNE Video Projection 1',
                                    't-SNE Video Projection 2',
                                    'Video 1',
                                    'Video Reconstruction AE 1',
                                    'Video 2',
                                    'Video Reconstruction AE 2']

                else:
                    metrics_list = [self.plt_loss,
                                    self.val_net_acc,
                                    pca_video_proj1,
                                    pca_video_proj2,
                                    lda_video_proj1,
                                    lda_video_proj2,
                                    tsne_video_proj1,
                                    tsne_video_proj2]
                    iterations_list = [list(range(1, (self.current_epoch *
                                                  self.num_batches)+1)),
                                       list(range(1, self.current_epoch+1)),
                                       None, None, None, None,
                                       None, None]
                    plt_types = ['lines', 'lines', 'scatter', 'scatter',
                                 'scatter', 'scatter', 'scatter',
                                 'scatter']
                    metric_names = ['Loss',
                                    'Network Accuracy',
                                    'PCA Video Projection 1',
                                    'PCA Video Projection 2',
                                    'LDA Video Projection 1',
                                    'LDA Video Projection 2',
                                    't-SNE Video Projection 1',
                                    't-SNE Video Projection 2']

                plot_metrics(
                    metrics_list=metrics_list,
                    iterations_list=iterations_list,
                    types=plt_types,
                    metric_names=metric_names,
                    legend=True,
                    # x_label='Iteration',
                    # y_label='Loss',
                    savefile=os.path.join(self.plt_dir,
                                          'metrics_epoch{}.png'
                                          .format(epoch)),
                    bot=self.bot)

                if self.plot_individually is True:
                    plot_metrics_individually(
                        metrics_list=metrics_list,
                        iterations_list=iterations_list,
                        types=plt_types,
                        metric_names=metric_names,
                        legend=True,
                        # x_label='Iteration',
                        # y_label='Loss',
                        savefile=os.path.join(self.plt_dir,
                                              'metrics_epoch{}.png'
                                              .format(epoch)),
                        bot=None)

                if self.svm_analysis is True:
                    plot_metrics(
                        metrics_list=plt_acc,
                        iterations_list=metrics_it_list,
                        types=metrics_plt_types,
                        metric_names=acc_plt_names,
                        legend=True,
                        # x_label='Iteration',
                        # y_label='Loss',
                        savefile=os.path.join(self.plt_dir,
                                              'acc_epoch{}.png'
                                              .format(epoch)),
                        bot=self.bot)
                    if self.plot_individually is True:
                        plot_metrics_individually(
                            metrics_list=plt_acc,
                            iterations_list=metrics_it_list,
                            types=metrics_plt_types,
                            metric_names=acc_plt_names,
                            legend=True,
                            # x_label='Iteration',
                            # y_label='Loss',
                            savefile=os.path.join(self.plt_dir,
                                                  'acc_epoch{}.png'
                                                  .format(epoch)),
                            bot=None)

                    plot_metrics(
                        metrics_list=plt_auc,
                        iterations_list=metrics_it_list,
                        types=metrics_plt_types,
                        metric_names=auc_plt_names,
                        legend=True,
                        # x_label='Iteration',
                        # y_label='Loss',
                        savefile=os.path.join(self.plt_dir,
                                              'auc_epoch{}.png'
                                              .format(epoch)),
                        bot=self.bot)

                    if self.plot_individually is True:
                        plot_metrics_individually(
                            metrics_list=plt_auc,
                            iterations_list=metrics_it_list,
                            types=metrics_plt_types,
                            metric_names=auc_plt_names,
                            legend=True,
                            # x_label='Iteration',
                            # y_label='Loss',
                            savefile=os.path.join(self.plt_dir,
                                                  'auc_epoch{}.png'
                                                  .format(epoch)),
                            bot=None)

            if epoch % self.checkpoint_epoch == 0 or \
                    epoch == 1 or \
                    epoch == self.epoch:
                n_epochs = self.checkpoint_epoch

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.plt_loss[-ii*self.num_batches:
                                      -(ii-1)*self.num_batches]
                        .reshape((1, -1)), index=[epoch-ii]) \
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
                        .reshape((1, -1)), index=[epoch-ii]) \
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
                        self.val_net_acc[-self.checkpoint_epoch:],
                        index=list(range(max(1, epoch-self.checkpoint_epoch+1),
                                         epoch+1))) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'val_net_acc',
                                append=True, format='table')

                if self.svm_analysis is True:
                    pd.DataFrame(
                        self.val_acc[-1].reshape((1, -1)), index=[epoch]) \
                        .to_hdf(
                            os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_acc',
                            append=True, format='table')

                    pd.DataFrame(
                        self.val_auc[-1].reshape((1, -1)), index=[epoch]) \
                        .to_hdf(
                            os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_auc',
                            append=True, format='table')

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

                if self.is_multilabel is True:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm is True:
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

                if self.is_multilabel is True:
                    inverted_labels = 1 - labels
                    labels = labels
                    inverted_labels = inverted_labels

                    labels = np.stack([inverted_labels, labels], axis=2)

                if self.use_batch_norm is True:
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


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    gpu_options = tf.GPUOptions(visible_device_list=args.gpu_id,
                                allow_growth=True)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1,
                                  gpu_options=gpu_options,
                                  allow_soft_placement=True)
    sess = tf.Session(config=session_conf)

    model = VideoRep(sess=sess,
                     model_id=args.model_id,
                     model_name=args.model_name,
                     model_type=args.model_type,
                     dataset_name=args.dataset_name,
                     class_velocity=args.class_velocity,
                     hide_digits=args.hide_digits,
                     tr_size=args.tr_size,
                     val_size=args.val_size,
                     use_batch_norm=args.use_batch_norm,
                     use_layer_norm=args.use_layer_norm,
                     use_l2_reg=args.use_l2_reg,
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
    else:
        raise NotImplementedError
