import subprocess

# KTH
configs = ['experiments/c3d_clf_exp201.nn.config',
           'experiments/r3d_clf_exp201.nn.config',
           'experiments/r21d_clf_exp201.nn.config',
           'experiments/cnn_lstm_exp201.nn.config',
           'experiments/cnn_lstm_exp204.nn.config',
           'experiments/cnn_lstm_exp207.nn.config']

for config in configs:
    print('Running -- {}'.format('python video_representations.py '
                                 'config_file --path {}'.format(config)))
    subprocess.call(['python', 'video_representations.py',
                     'config_file', '--path', '{}'.format(config)])
