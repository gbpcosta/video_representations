import subprocess

# UCF
configs = ['experiments/c3d_clf_exp301.nn.config',
           'experiments/r3d_clf_exp301.nn.config',
           'experiments/r21d_clf_exp301.nn.config',
           'experiments/cnn_lstm_exp301.nn.config',
           'experiments/cnn_lstm_exp304.nn.config',
           'experiments/cnn_lstm_exp307.nn.config']

for config in configs:
    print('Running -- {}'.format('python video_representations.py '
                                 'config_file --path {}'.format(config)))
    subprocess.call(['python', 'video_representations.py',
                     'config_file', '--path', '{}'.format(config)])
