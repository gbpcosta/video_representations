import os
import argparse
import configparser
import numpy as np
from utils import check_folder


def parse_args():
    """ Parsing and configuration """

    desc = 'Neural network training leveraging semantic information'
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(dest='command',
                                       help='Choice between using '
                                            'configuration file and command '
                                            ' line arguments.')
    # group = parser.add_mutually_exclusive_group(required=True)

    # create the parser for the "config_file" command
    parser_a = subparsers.add_parser('config_file',
                                     help='Read parameters from config file.')
    parser_a.add_argument('--path', type=str,
                          help='Path to configuration file.', required=True)

    # create the parser for the "args" command
    # parser_b = group.add_argument_group(title='inline arguments')
    parser_b = subparsers.add_parser('args',
                                     help='Read parameters from command'
                                          ' line.')

    # MODEL
    parser_b.add_argument('--model_id', type=int, default=-1,
                          help='ID for model')
    parser_b.add_argument('--model_name', type=str, default='',
                          help='Optional string that will be appended to the '
                               'model name to help identify plots and '
                               'checkpoints')
    parser_b.add_argument('--model_type', type=str, default='c3d_ae_small',
                          choices=['c3d_ae_small', 'c3d_clf_small',
                                   'c3d_ae_large', 'c3d_clf_large',
                                   'lstm_ae_small', 'lstm_ae_large',
                                   'gru_ae_small', 'gru_ae_large',
                                   'p3d_ae_small', 'p3d_ae_large',
                                   'r3d_clf_small', 'r3d_clf_large',
                                   'r21d_clf_small', 'r21d_clf_large'],
                          help='Architecture type used for the network',
                          required=True)
    parser_b.add_argument('--ex_mode', type=str, default='training',
                          choices=['training', 'feat_extraction'],
                          help='Execution mode: training or feature '
                               'extraction.',
                          required=True)

    # DATASET
    parser_b.add_argument('--dataset_name', type=str, default='bouncingMNIST',
                          choices=['bouncingMNIST'],
                          help='Dataset used for training and validation',
                          required=True)
    parser_b.add_argument('--tr_size', type=np.int64, default=12800,
                          help='Size of training dataset')
    parser_b.add_argument('--val_size', type=np.int64, default=5120,
                          help='Size of validation dataset')

    # NETWORK ARCHITECTURE
    parser_b.add_argument('--use_batch_norm', action='store_true',
                          help='Activates batch normalization')
    parser_b.add_argument('--use_layer_norm', action='store_true',
                          help='Activates layer normalization')
    parser_b.add_argument('--use_l2_reg', action='store_true',
                          help='Activates l2 regularization')

    # TRAINING
    parser_b.add_argument('--epoch', type=np.int64, default=100,
                          help='Number of epochs to run')
    parser_b.add_argument('--batch_size', type=np.int64, default=32,
                          help='Size of minibatch')
    parser_b.add_argument('--learning_rate', type=np.float32, default=10e-4,
                          help='Learning rate used for the optimizer')

    # VISUALIZATION & ANALYSIS
    parser_b.add_argument('--svm_analysis', action='store_true',
                          help='Activates svm analysis on embedding space')
    parser_b.add_argument('--vis_epoch', type=np.int64, default=10,
                          help='Number of epochs before computing and plotting'
                               ' evaluation metrics and visualizations')
    parser_b.add_argument('--plot_individually', action='store_true',
                          help='Creates a separate plot for each metric/'
                               'visualization.')
    parser_b.add_argument('-v', '--verbosity',
                          action='count', default=0,
                          help='Controls output verbosity')

    # CHECKPOINT
    parser_b.add_argument('--checkpoint_epoch', type=np.int64, default=10,
                          help='Save checkpoint every <checkpoint_epoch> '
                               'epochs')
    parser_b.add_argument('--keep_checkpoint_max', type=np.int64, default=1,
                          help='Number of checkpoints to keep on disk')
    parser_b.add_argument('--redo', action='store_true',
                          help='Redo training from the start regardless of'
                               'finding checkpoint')

    # INTEGRATION
    parser_b.add_argument('--slack_bot', action='store_true',
                          help='Activate Slack bot that sends generated plots '
                               'to Slack channels specified on slack.config')

    # DIRECTORIES
    parser_b.add_argument('--plt_dir', type=str,
                          default='_outputs/{}/plots',
                          help='Path to directory used to save plots')
    parser_b.add_argument('--model_dir', type=str,
                          default='_outputs/{}/models',
                          help='Path to directory used to save trained models')
    parser_b.add_argument('--config_dir', type=str,
                          default='_outputs/{}/configs',
                          help='Path to directory used to save trained '
                               'configuration files')

    # HARDWARE
    parser_b.add_argument('--gpu_id', type=str, default='0',
                          help='GPU ID used to run the experiment')

    args = parser.parse_args()
    if args.command == 'config_file':
        args = read_config_file(args.path)
    else:
        args = parser.parse_args()

    return check_args(args)


def read_config_file(path):
    config = configparser.ConfigParser()
    config.read(path)

    class Arguments(object):

        default = {'model_id': -1,
                   'model_name': '',
                   'model_type': 'c3d_ae_small',
                   'ex_mode': 'training',
                   'dataset_name': 'bouncingMNIST',
                   'tr_size': 12800,
                   'val_size': 5120,
                   'use_batch_norm': False,
                   'use_layer_norm': False,
                   'use_l2_reg': False,
                   'epoch': 100,
                   'batch_size': 32,
                   'learning_rate': 10e-4,
                   'svm_analysis': False,
                   'vis_epoch': 10,
                   'plot_individually': False,
                   'verbosity': 0,
                   'checkpoint_epoch': 10,
                   'keep_checkpoint_max': 1,
                   'redo': False,
                   'slack_bot': True,
                   'plt_dir': '_outputs/{}/plots',
                   'model_dir': '_outputs/{}/models',
                   'config_dir': '_outputs/{}/configs',
                   'gpu_id': '0'}

        def __init__(self, config, parse=True):
            if parse is True:
                def parse_value_or_get_default(get_func, section, option):
                    value_set = config.has_option(section, option)

                    if value_set is True:
                        return get_func(section, option)
                    else:
                        return self.default[option]

                self.model_id = parse_value_or_get_default(config.getint,
                                                           'MODEL',
                                                           'model_id')
                self.model_name = parse_value_or_get_default(config.get,
                                                             'MODEL',
                                                             'model_name')
                self.model_type = parse_value_or_get_default(config.get,
                                                             'MODEL',
                                                             'model_type')
                self.ex_mode = parse_value_or_get_default(config.get,
                                                          'MODEL',
                                                          'ex_mode')

                # DATASET
                self.dataset_name = parse_value_or_get_default(config.get,
                                                               'DATASET',
                                                               'dataset_name')
                self.tr_size = parse_value_or_get_default(config.getint,
                                                          'DATASET', 'tr_size')
                self.val_size = parse_value_or_get_default(config.getint,
                                                           'DATASET',
                                                           'val_size')

                # NETWORK ARCHITECTURE
                self.use_batch_norm = \
                    parse_value_or_get_default(config.getboolean,
                                               'NETWORK ARCHITECTURE',
                                               'use_batch_norm')
                self.use_layer_norm = \
                    parse_value_or_get_default(config.getboolean,
                                               'NETWORK ARCHITECTURE',
                                               'use_layer_norm')
                self.use_l2_reg = \
                    parse_value_or_get_default(config.getboolean,
                                               'NETWORK ARCHITECTURE',
                                               'use_l2_reg')

                # TRAINING
                self.epoch = parse_value_or_get_default(config.getint,
                                                        'TRAINING',
                                                        'epoch')
                self.batch_size = parse_value_or_get_default(config.getint,
                                                             'TRAINING',
                                                             'batch_size')
                self.learning_rate = \
                    parse_value_or_get_default(config.getfloat,
                                               'TRAINING',
                                               'learning_rate')

                # VISUALIZATION & ANALYSIS
                self.svm_analysis = \
                    parse_value_or_get_default(config.getboolean,
                                               'VISUALIZATION',
                                               'svm_analysis')
                self.vis_epoch = parse_value_or_get_default(config.getint,
                                                            'VISUALIZATION',
                                                            'vis_epoch')
                self.plot_individually = \
                    parse_value_or_get_default(config.getboolean,
                                               'VISUALIZATION',
                                               'plot_individually')
                self.verbosity = parse_value_or_get_default(config.getint,
                                                            'VISUALIZATION',
                                                            'verbosity')

                # CHECKPOINT
                self.checkpoint_epoch = \
                    parse_value_or_get_default(config.getint, 'CHECKPOINT',
                                               'checkpoint_epoch')
                self.keep_checkpoint_max = \
                    parse_value_or_get_default(config.getint, 'CHECKPOINT',
                                               'keep_checkpoint_max')
                self.redo = parse_value_or_get_default(config.getboolean,
                                                       'CHECKPOINT', 'redo')

                # INTEGRATION
                self.slack_bot = parse_value_or_get_default(config.getboolean,
                                                            'INTEGRATION',
                                                            'slack_bot')

                # DIRECTORIES
                self.plt_dir = parse_value_or_get_default(config.get,
                                                          'DIRECTORIES',
                                                          'plt_dir')
                self.model_dir = parse_value_or_get_default(config.get,
                                                            'DIRECTORIES',
                                                            'model_dir')
                self.config_dir = parse_value_or_get_default(config.get,
                                                             'DIRECTORIES',
                                                             'config_dir')

                # HARDWARE
                self.gpu_id = parse_value_or_get_default(config.get,
                                                         'HARDWARE', 'gpu_id')
            else:
                self.model_id = args.model_id
                self.model_name = args.model_name
                self.model_type = args.model_type
                self.ex_mode = args.ex_mode

                # DATASET
                self.dataset_name = args.dataset_name
                self.tr_size = args.tr_size
                self.val_size = args.val_size

                # NETWORK ARCHITECTURE
                self.use_batch_norm = args.use_batch_norm
                self.use_layer_norm = args.use_layer_norm
                self.use_l2_reg = args.use_l2_reg

                # TRAINING
                self.epoch = args.epoch
                self.batch_size = args.batch_size
                self.learning_rate = args.learning_rate

                # VISUALIZATION
                self.svm_analysis = args.svm_analysis
                self.vis_epoch = args.vis_epoch
                self.plot_individually = self.plot_individually
                self.verbosity = args.verbosity

                # CHECKPOINT
                self.checkpoint_epoch = args.checkpoint_epoch
                self.keep_checkpoint_max = args.keep_checkpoint_max
                self.redo = args.redo

                # INTEGRATION
                self.slack_bot = args.slack_bot

                # DIRECTORIES
                self.plt_dir = args.plt_dir
                self.model_dir = args.config_dir
                self.config_dir = args.config_dir

                # HARDWARE
                self.gpu_id = args.gpu_id

        def generate_config_file(self, path):
            self.config = configparser.ConfigParser()

            self.config['MODEL'] = {'model_id': self.model_id,
                                    'model_name': self.model_name,
                                    'model_type': self.model_type,
                                    'ex_mode': self.ex_mode}

            self.config['DATASET'] = {'dataset_name': self.dataset_name,
                                      'tr_size': self.tr_size,
                                      'val_size': self.val_size}

            self.config['NETWORK ARCHITECTURE'] = \
                {'use_batch_norm': self.use_batch_norm,
                 'use_layer_norm': self.use_layer_norm,
                 'use_l2_reg': self.use_l2_reg}


            self.config['TRAINING'] = {'epoch': self.epoch,
                                       'batch_size': self.batch_size,
                                       'learning_rate': self.learning_rate}

            self.config['VISUALIZATION'] = \
                {'svm_analysis': self.svm_analysis,
                 'vis_epoch': self.vis_epoch,
                 'plot_individually': self.plot_individually,
                 'verbosity': self.verbosity}

            self.config['CHECKPOINT'] = \
                {'checkpoint_epoch': self.checkpoint_epoch,
                 'keep_checkpoint_max': self.keep_checkpoint_max,
                 'redo': self.redo}

            self.config['INTEGRATION'] = {'slack_bot': self.slack_bot}

            self.config['DIRECTORIES'] = {'plt_dir': self.plt_dir,
                                          'model_dir': self.model_dir,
                                          'config_dir': self.config_dir}

            self.config['HARDWARE'] = {'gpu_id': self.gpu_id}

            self.config.write(open(path, 'w'))

    args = Arguments(config)

    return args


def check_args(args):
    """checking arguments"""

    # MODEL
    # assert args.model_id
    assert args.model_type in ['c3d_ae_small', 'c3d_clf_small',
                               'c3d_ae_large', 'c3d_clf_large',
                               'lstm_ae_small',
                               'lstm_ae_large', 'gru_ae_small', 'gru_ae_large',
                               'p3d_ae_small', 'p3d_ae_large',
                               'r3d_clf_small', 'r3d_clf_large',
                               'r21d_clf_small', 'r21d_clf_large'], \
        'invalid model_type. model_type must be one of the following: ' \
        'c3d_ae_small, c3d_ae_large, ' \
        'lstm_ae_small, lstm_ae_large, ' \
        'gru_ae_small, gru_ae_large, ' \
        'p3d_ae_small, p3d_ae_large, ' \
        'r3d_clf_small, r3d_clf_large, ' \
        'r21d_clf_small, r21d_clf_large.'

    assert args.ex_mode in ['training', 'feat_extraction'], \
        'invalid ex_mode. ex_mode must be one of the following: ' \
        'training, feat_extraction.' \

    # DATASET
    assert args.dataset_name in ['bouncingMNIST'], \
        'invalid dataset_name. dataset_name must be one of the following: ' \
        'bouncingMNIST.'
    assert args.tr_size > 0, 'tr_size must be greater than 0.'
    assert args.val_size > 0, 'val_size must be greater than 0.'

    # NETWORK ARCHITECTURE

    # TRAINING
    assert args.epoch > 0, \
        'pretrain_ae_epoch must be greater than 0.'
    assert args.batch_size > 0, \
        'batch_size must be greater than 0.'
    assert args.batch_size <= args.tr_size, \
        'batch_size must be smaller or equal to tr_size'
    assert args.batch_size <= args.val_size, \
        'batch_size must be smaller or equal to val_size'
    assert args.learning_rate > 0, \
        'pretrain_sem_ae_lr must be greater than 0.'

    # LOSS

    # VISUALIZATION
    assert args.vis_epoch > 0, 'vis_epoch must be greater than 0.'

    # CHECKPOINT
    assert args.checkpoint_epoch > 0, \
        'checkpoint_epoch must be greater than 0.'
    assert args.keep_checkpoint_max >= 0, \
        'keep_checkpoint_max must be greater or equal to 0.'

    # INTEGRATION

    # DIRECTORIES
    # --plt_dir
    check_folder(args.plt_dir.format(
        '{}_{}'.format(args.model_name, args.model_id)))
    # --model_dir
    check_folder(args.model_dir.format(
        '{}_{}'.format(args.model_name, args.model_id)))
    # --config_dir
    check_folder(args.config_dir.format(
        '{}_{}'.format(args.model_name, args.model_id)))
    # save config file to config_dir
    args.generate_config_file(
        os.path.join(args.config_dir.format(
            '{}_{}'.format(args.model_name, args.model_id)),
            'nn.config'))

    return args
