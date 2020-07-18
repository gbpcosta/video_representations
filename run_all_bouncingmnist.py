import subprocess

# BouncingMNIST
configs = [# 'experiments/c3d_clf_exp101.nn.config',
          #  'experiments/c3d_clf_exp101.1.nn.config',
          #  'experiments/c3d_clf_exp102.nn.config',
          #  'experiments/c3d_clf_exp102.1.nn.config',
          #  'experiments/c3d_clf_exp103.nn.config',
          #  'experiments/c3d_clf_exp103.1.nn.config',
          #  'experiments/r3d_clf_exp101.nn.config',
          #  'experiments/r3d_clf_exp101.1.nn.config',
          #  'experiments/r3d_clf_exp102.nn.config',
          #  'experiments/r3d_clf_exp102.1.nn.config',
          #  'experiments/r3d_clf_exp103.nn.config',
          #  'experiments/r3d_clf_exp103.1.nn.config',
          #  'experiments/r21d_clf_exp101.nn.config',
          #  'experiments/r21d_clf_exp101.1.nn.config',
          #  'experiments/r21d_clf_exp102.nn.config',
          #  'experiments/r21d_clf_exp102.1.nn.config',
          #  'experiments/r21d_clf_exp103.nn.config',
          #  'experiments/r21d_clf_exp103.1.nn.config',
          #  'experiments/cnn_lstm_exp101.nn.config',
          #  'experiments/cnn_lstm_exp101.1.nn.config',
           'experiments/cnn_lstm_exp102.nn.config',
           'experiments/cnn_lstm_exp102.1.nn.config',
           'experiments/cnn_lstm_exp103.nn.config',
           'experiments/cnn_lstm_exp103.1.nn.config',
           'experiments/cnn_lstm_exp104.nn.config',
           'experiments/cnn_lstm_exp104.1.nn.config',
           'experiments/cnn_lstm_exp105.nn.config',
           'experiments/cnn_lstm_exp105.1.nn.config',
           'experiments/cnn_lstm_exp106.nn.config',
           'experiments/cnn_lstm_exp106.1.nn.config']

for config in configs:
    print('Running -- {}'.format('python video_representations.py '
                                 'config_file --path {}'.format(config)))
    subprocess.call(['python', 'video_representations.py',
                     'config_file', '--path', '{}'.format(config)])
