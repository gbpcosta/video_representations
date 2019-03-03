import subprocess

# BouncingMNIST
# configs = ['c3d_clf_exp101.nn.config',
#            'c3d_clf_exp102.nn.config',
#            'c3d_clf_exp102.1.nn.config',
#            'c3d_clf_exp103.nn.config',
#            'c3d_clf_exp103.1.nn.config',
#            'r3d_clf_exp101.nn.config',
#            'r3d_clf_exp102.nn.config',
#            'r3d_clf_exp102.1.nn.config',
#            'r3d_clf_exp103.nn.config',
#            'r3d_clf_exp103.1.nn.config',
#            'r21d_clf_exp101.nn.config',
#            'r21d_clf_exp102.nn.config',
#            'r21d_clf_exp102.1.nn.config',
#            'r21d_clf_exp103.nn.config',
#            'r21d_clf_exp103.1.nn.config',
#            'cnn_lstm_exp101.nn.config',
#            'cnn_lstm_exp102.nn.config',
#            'cnn_lstm_exp102.1.nn.config',
#            'cnn_lstm_exp103.nn.config',
#            'cnn_lstm_exp103.1.nn.config',
#            'cnn_lstm_exp104.nn.config',
#            'cnn_lstm_exp105.nn.config',
#            'cnn_lstm_exp105.1.nn.config',
#            'cnn_lstm_exp106.nn.config',
#            'cnn_lstm_exp106.1.nn.config'
#            'cnn_lstm_exp107.nn.config',
#            'cnn_lstm_exp108.nn.config',
#            'cnn_lstm_exp108.1.nn.config',
#            'cnn_lstm_exp109.nn.config',
#            'cnn_lstm_exp109.1.nn.config']

# KTH
# configs = ['c3d_clf_exp201.nn.config',
#            'r3d_clf_exp201.nn.config',
#            'r21d_clf_exp201.nn.config',
#            'cnn_lstm_exp201.nn.config',
#            'cnn_lstm_exp204.nn.config',
#            'cnn_lstm_exp207.nn.config']

# UCF
configs = ['c3d_clf_exp301.nn.config',
           'r3d_clf_exp301.nn.config',
           'r21d_clf_exp301.nn.config',
           'cnn_lstm_exp301.nn.config',
           'cnn_lstm_exp304.nn.config',
           'cnn_lstm_exp307.nn.config']

for config in configs:
    print('Running -- {}'.format('python video_representations.py '
                                 'config_file --path {}'.format(config)))
    subprocess.call(['python', 'video_representations.py',
                     'config_file', '--path', '{}'.format(config)])
