import tensorflow as tf
from ConvRNNCell import ConvLSTMCell
tl = tf.layers

def def_convlstm_small_video_classifier(input, is_training=True, reuse=False,
                            use_l2_reg=False,
                            use_batch_norm=False, use_layer_norm=False,
                            video_ae_emb_layer_name='',
                            emb_dim=1024):
        if use_l2_reg:
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        else:
            regularizer = None

        initializer = tf.glorot_normal_initializer()

        with tf.variable_scope('video_ae_encoder', reuse=reuse) as vs:
            def convlstm_cell(shape, filters):  # , initializer=initializer):
                return  ConvLSTMCell(shape=shape,
                                     filters=filters,
                                     kernel=(3, 3),
                                     forget_bias=1.0,
                                     activation=tf.nn.relu,
                                     normalize=True,
                                     peephole=True,
                                     data_format='channels_last')
                # return tf.contrib.rnn.ConvLSTMCell(
                #     conv_ndims=2,
                #     input_shape=tf.shape(input)[1:],
                #     output_channels=n_kernels,
                #     kernel_shape=(3,3),
                #     skip_connection=False,
                #     initializers=initializer,
                #     name='conv_lstm_cell')


            stacked_convlstm = \
                tf.nn.rnn_cell.MultiRNNCell(
                    cells=[convlstm_cell(shape=tf.shape(input)[2:4],
                                         filters=32),
                           convlstm_cell(shape=tf.shape(input)[2:4],
                                         filters=32),
                           convlstm_cell(shape=tf.shape(input)[2:4],
                                         filters=64),
                           convlstm_cell(shape=tf.shape(input)[2:4],
                                         filters=64),
                           convlstm_cell(shape=tf.shape(input)[2:4],
                                         filters=128)],
                    state_is_tuple=True)

            output, state = tf.nn.dynamic_rnn(cell=stacked_convlstm,
                                              inputs=input,
                                              dtype=inputs.dtype)

            video_ae_de_out = tf.reshape(tf.stack(output, axis=1),
                                         [-1, 2048])

            video_ae_de_out = tl.dense(video_ae_de_out,
                                       units=4096,
                                       activation=None,
                                       name='vde_fc1')

            video_ae_recon_logits = tf.reshape(video_ae_de_out,
                                               [-1, input.shape[1], 64, 64, 1])
            video_ae_recon = tf.nn.sigmoid(video_ae_recon_logits)

        video_ae_vars = tf.contrib.framework.get_variables(vs) + \
            tf.contrib.framework.get_variables(vs2) + \
            tf.contrib.framework.get_variables(vs3)

        # get last embedding for each video
        video_ae_emb = tf.reshape(video_ae_emb,
                                  [-1, input.shape[1],
                                   emb_dim])
        video_ae_emb = video_ae_emb[:, -1, :]

        return video_ae_recon, video_ae_recon_logits, \
            video_ae_emb, video_ae_vars


def def_lstm_small_video_ae(input, is_training=True, reuse=False,
                            use_l2_reg=False,
                            use_batch_norm=False, use_layer_norm=False,
                            video_ae_emb_layer_name='video_ae_emb_layer',
                            emb_dim=1024):
    """ VIDEO AE """

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    initializer = tf.glorot_normal_initializer()

    with tf.variable_scope('video_ae_encoder', reuse=reuse) as vs:
        video_ae_en_cell = tf.nn.rnn_cell.LSTMCell(num_units=2048,
                                                   use_peepholes=True,
                                                   cell_clip=None,
                                                   initializer=initializer,
                                                   num_proj=None,
                                                   proj_clip=None,
                                                   num_unit_shards=None,
                                                   num_proj_shards=None,
                                                   forget_bias=1.0,
                                                   state_is_tuple=True,
                                                   activation=tf.nn.relu,
                                                   name="ven_lstm1")

        input_en = tf.reshape(input,
                              [-1,                        # batch size
                               input.shape[1],            # sequence length
                               64*64*1])  # dims
        input_en = tf.unstack(input_en, num=input.shape[1], axis=1)

        video_ae_en_out, video_ae_en_state = \
            tf.nn.static_rnn(cell=video_ae_en_cell,
                             inputs=input_en,
                             # sequence_length=np.repeat(16, batch_size),
                             dtype=tf.float32)

        video_ae_en_out = tf.reshape(tf.stack(video_ae_en_out, axis=1),
                                     [-1, 2048])

        if use_batch_norm is True:
            video_ae_en_out = \
                tl.batch_normalization(inputs=video_ae_en_out,
                                       training=is_training,
                                       name='ven_bn1')
        if use_layer_norm is True:
            video_ae_en_out = tf.contrib.layers.layer_norm(
                inputs=video_ae_en_out)

    with tf.variable_scope(video_ae_emb_layer_name, reuse=reuse) as \
            vs2:
        """ CODE """
        video_ae_emb = tl.dense(video_ae_en_out,
                                units=emb_dim,
                                activation=tf.nn.relu,
                                name='emb_space')

        video_ae_emb_out = tf.unstack(tf.reshape(video_ae_emb,
                                                 [-1, input.shape[1],
                                                  emb_dim]),
                                      axis=1)

    with tf.variable_scope('video_ae_decoder', reuse=reuse) as vs3:
        video_ae_de_cell = tf.nn.rnn_cell.LSTMCell(num_units=2048,
                                                   use_peepholes=True,
                                                   cell_clip=None,
                                                   initializer=initializer,
                                                   num_proj=None,
                                                   proj_clip=None,
                                                   num_unit_shards=None,
                                                   num_proj_shards=None,
                                                   forget_bias=1.0,
                                                   state_is_tuple=True,
                                                   activation=None,
                                                   name="vde_lstm1")

        video_ae_de_out, video_ae_de_state = \
            tf.nn.static_rnn(cell=video_ae_de_cell,
                             inputs=video_ae_emb_out,
                             initial_state=video_ae_en_state,
                             # sequence_length=np.repeat(16, batch_size),
                             dtype=tf.float32)

        if use_batch_norm is True:
            video_ae_de_cell = \
                tl.batch_normalization(inputs=video_ae_de_out,
                                       training=is_training,
                                       name='vde_bn1')
        if use_layer_norm is True:
            video_ae_de_cell = tf.contrib.layers.layer_norm(
                inputs=video_ae_de_out)

        video_ae_de_out = video_ae_de_out[::-1]

        # video_ae_de_out = tf.stack(video_ae_de_out, axis=1)

        video_ae_de_out = tf.reshape(tf.stack(video_ae_de_out, axis=1),
                                     [-1, 2048])

        video_ae_de_out = tl.dense(video_ae_de_out,
                                   units=4096,
                                   activation=None,
                                   name='vde_fc1')

        video_ae_recon_logits = tf.reshape(video_ae_de_out,
                                           [-1, input.shape[1], 64, 64, 1])
        video_ae_recon = tf.nn.sigmoid(video_ae_recon_logits)

    video_ae_vars = tf.contrib.framework.get_variables(vs) + \
        tf.contrib.framework.get_variables(vs2) + \
        tf.contrib.framework.get_variables(vs3)

    # get last embedding for each video
    video_ae_emb = tf.reshape(video_ae_emb,
                              [-1, input.shape[1],
                               emb_dim])
    video_ae_emb = video_ae_emb[:, -1, :]

    return video_ae_recon, video_ae_recon_logits, \
        video_ae_emb, video_ae_vars


def def_lstm_large_video_ae(input, is_training=True, reuse=False,
                            use_l2_reg=False,
                            use_batch_norm=False, use_layer_norm=False,
                            video_ae_emb_layer_name='video_ae_emb_layer',
                            emb_dim=1024):
    raise NotImplementedError


def def_lstm_small_video_classifier(
        input, is_training=True, reuse=False,
        use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_ae_emb_layer_name='clf_fc6'):
    raise NotImplementedError


def def_lstm_large_video_classifier(
        input, is_training=True, reuse=False,
        use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_ae_emb_layer_name='clf_fc6'):
    raise NotImplementedError
