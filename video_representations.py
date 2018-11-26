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

from models_c3d import def_c3d_large_video_ae, \
                       def_c3d_small_video_ae, \
                       def_c3d_large_video_classifier, \
                       def_c3d_small_video_classifier
from models_lstm import def_lstm_large_video_ae, def_lstm_small_video_ae
from models_gru import def_gru_large_video_ae, def_gru_small_video_ae

from bouncingMNIST import BouncingMNISTDataGenerator
from utils import plot_metrics

tl = tf.layers

os.environ["PYTHONHASHSEED"] = '42'
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)


class VideoRep():

    def __init__(self, sess,
                 model_id, model_name, model_type,
                 dataset_name, tr_size, val_size,
                 use_batch_norm, use_layer_norm, use_l2_reg,
                 epoch, batch_size, learning_rate,
                 vis_epoch, verbosity,
                 checkpoint_epoch, keep_checkpoint_max, redo,
                 slack_bot,
                 plt_dir, model_dir, config_dir):
        self.sess = sess

        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.is_ae = ('ae' in self.model_type)

        if 'small' in self.model_type:
            self.emb_dim = 1024
        elif 'large' in self.model_type:
            self.emb_dim = 2048

        self.dataset_name = dataset_name
        self.tr_size = tr_size
        self.val_size = val_size

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_l2_reg = use_l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = 0.9

        self.vis_epoch = vis_epoch
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

        self.training_generator = \
            BouncingMNISTDataGenerator(dataset_size=self.tr_size,
                                       batch_size=self.batch_size,
                                       ae=self.is_ae,
                                       noise=self.is_ae)
        self.validation_generator = \
            BouncingMNISTDataGenerator(dataset_size=self.val_size,
                                       batch_size=self.batch_size,
                                       ae=self.is_ae,
                                       split='test')

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

                self.current_epoch = (self.plt_loss.shape[0] //
                                      self.num_batches) + 1
                if self.verbosity >= 1:
                    print('Checkpoint loaded - epoch {}'
                          .format(self.current_epoch))

        if self.current_epoch == 1:
            self.plt_loss = np.array([])
            self.val_loss = np.array([])

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
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.net_out_logits,
                    labels=self.labels))

        if self.use_batch_norm is True:
            if self.is_ae:  # mse
                self.test_loss = tf.reduce_mean(
                    tf.square(self.test_net_out - self.test_video))

            else:  # binary_crossentropy
                self.test_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.test_net_out_logits,
                        labels=self.test_labels))

    def _def_optimizer(self):
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            optim = \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                       beta1=self.beta1) \
                .minimize(self.loss,
                          var_list=self.learnable_vars)

            return optim

    def _def_svm_clf():
        pass

    def _def_metrics():
        if self.is_ae is True:
            pass
        else:
            if self.dataset_name == 'bouncingMNIST':
                self.acc, self.acc_op = tf.metrics.mean_per_class_accuracy(
                        labels=self.labels,
                        predictions=self.net_out_logits,
                        num_classes=2)
            else:
                self.acc, self.acc_op = tf.metrics.accuracy(
                    labels=tf.argmax(self.labels, axis=1),
                    predictions=tf.argmax(self.net_out_logits, axis=1))

            self.auc, self.auc_op = tf.metrics.auc(
                labels=self.labels,
                predictions=self.net_out_logits)

    def _def_model(self):
        """ INPUTS """
        self.video = tf.placeholder(tf.float32, [None, 16, 64, 64, 1])
        self.labels = tf.placeholder(tf.float32, [None, 10])

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
                    self.video,
                    is_training=True,
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
                    self.video,
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
                    self.video,
                    is_training=True,
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

        if self.use_batch_norm is True:
            self.test_video = tf.placeholder(tf.float32, [None, 16, 64, 64, 1])
            self.test_labels = tf.placeholder(tf.float32, [None, 10])

            if self.model_type == 'c3d_ae_small':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_c3d_small_video_ae(self.video,
                                           is_training=False,
                                           reuse=True)
            elif self.model_type == 'c3d_ae_large':
                self.test_net_out, \
                    self.test_net_out_logits, \
                    self.test_video_emb, \
                    self.test_learnable_vars, \
                    self.emb_dim = \
                    def_c3d_large_video_ae(self.video,
                                           is_training=False,
                                           reuse=True)
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

        self._def_loss_fn()
        self._def_metrics()
        self.optim = self._def_optimizer()

    def train_model(self):
        if self.current_epoch == 1:
            self.sess.run(tf.global_variables_initializer())

        start_time = time.time()

        starting_epoch = self.current_epoch

        for epoch in tqdm(range(starting_epoch, self.epoch+1), position=1):
            # Training
            for batch_number in tqdm(range(1, self.num_batches+1),
                                     position=0):
                video_batch, labels = \
                    self.training_generator.get_batch()

                _, _, _, loss, acc, auc, net_out = \
                    self.sess.run([self.optim, self.acc_op, self.auc_op,
                                   self.loss, self.acc, self.auc,
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

            # Validation and Visualization
            val_video_emb = np.array([], dtype=np.float32) \
                .reshape(0, self.emb_dim)
            val_labels = np.array([], dtype=np.float32) \
                .reshape(0, 10)
            for batch_number in tqdm(range(self.val_num_batches), position=0):
                video_batch, labels = \
                    self.validation_generator.get_batch()

                if self.use_batch_norm is True:
                    loss, net_out, video_emb = \
                        self.sess.run([self.test_loss,
                                       self.test_net_out,
                                       self.test_video_emb],
                                      feed_dict={self.test_video: video_batch,
                                                 self.test_labels: labels})
                else:
                    loss, net_out, video_emb = \
                        self.sess.run([self.loss, self.net_out,
                                       self.video_emb],
                                      feed_dict={self.video: video_batch,
                                                 self.labels: labels})

                self.val_loss = np.append(self.val_loss, loss)
                val_video_emb = np.vstack([val_video_emb, video_emb])
                val_labels = np.vstack([val_labels, labels])
            self.validation_generator.on_epoch_end()

            def get_labels(one_hot_labels, n_labels=2):
                labels = [np.nonzero(one_hot_labels[ii, :])[0]
                          for ii in range(one_hot_labels.shape[0])]
                labels = [np.repeat(labels[ii], n_labels).reshape(1, -1)
                          if labels[ii].shape[0] < n_labels
                          else labels[ii].reshape(1, -1)
                          for ii in range(len(labels))]

                return np.concatenate(labels, axis=0)

            if epoch % self.vis_epoch == 0 or \
                    epoch == 1 or \
                    epoch == self.epoch:
                val_labels_n = get_labels(val_labels)
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
                    iterations_list = [list(range(self.current_epoch *
                                                  self.num_batches)),
                                       None, None, None, None,
                                       None, None, None,
                                       None, None, None]
                    plt_types = ['lines', 'scatter', 'scatter',
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
                                    pca_video_proj1,
                                    pca_video_proj2,
                                    lda_video_proj1,
                                    lda_video_proj2,
                                    tsne_video_proj1,
                                    tsne_video_proj2]
                    iterations_list = [list(range(self.current_epoch *
                                                  self.num_batches)),
                                       None, None, None, None,
                                       None, None]
                    plt_types = ['lines', 'scatter', 'scatter',
                                 'scatter', 'scatter', 'scatter',
                                 'scatter']
                    metric_names = ['Loss',
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

            if epoch % self.checkpoint_epoch == 0 or \
                    epoch == self.epoch:
                self.saver.save(
                    self.sess,
                    os.path.join(
                        self.model_dir,
                        'checkpoint-epoch'),
                    global_step=epoch)

                n_epochs = self.checkpoint_epoch

                overall_epoch = epoch

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.plt_loss[-ii*self.num_batches:
                                      -(ii-1)*self.num_batches]
                        .reshape((1, -1)), index=[overall_epoch-ii]) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'loss',
                                append=True, format='table')

                pd.DataFrame(
                    self.plt_loss[-self.num_batches:]
                    .reshape((1, -1)), index=[overall_epoch]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'loss',
                            append=True, format='table')

                for ii in range(n_epochs, 1, -1):
                    pd.DataFrame(
                        self.val_loss[-ii*self.val_num_batches:
                                      -(ii-1)*self.val_num_batches]
                        .reshape((1, -1)), index=[overall_epoch-ii]) \
                        .to_hdf(os.path.join(self.model_dir,
                                             'plt_loss_bkup.h5'),
                                'val_loss',
                                append=True, format='table')

                pd.DataFrame(
                    self.val_loss[-self.val_num_batches:]
                    .reshape((1, -1)), index=[overall_epoch]) \
                    .to_hdf(os.path.join(self.model_dir,
                                         'plt_loss_bkup.h5'),
                            'val_loss',
                            append=True, format='table')

            self.training_generator.on_epoch_end()
            self.current_epoch += 1


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
                     tr_size=args.tr_size,
                     val_size=args.val_size,
                     use_batch_norm=args.use_batch_norm,
                     use_layer_norm=args.use_layer_norm,
                     use_l2_reg=args.use_l2_reg,
                     epoch=args.epoch,
                     batch_size=args.batch_size,
                     learning_rate=args.learning_rate,
                     vis_epoch=args.vis_epoch,
                     verbosity=args.verbosity,
                     checkpoint_epoch=args.checkpoint_epoch,
                     keep_checkpoint_max=args.keep_checkpoint_max,
                     redo=args.redo,
                     slack_bot=args.slack_bot,
                     plt_dir=args.plt_dir,
                     model_dir=args.model_dir,
                     config_dir=args.config_dir)
    model.train_model()
