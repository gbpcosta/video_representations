import tensorflow as tf
tl = tf.layers


def def_cnnlstm_small_video_classifier(inputs,
                                       n_classes,
                                       is_training=True,
                                       is_multilabel=False,
                                       reuse=False,
                                       use_l2_reg=False,
                                       use_batch_norm=False,
                                       use_layer_norm=False,
                                       video_emb_layer_name='lstm_out'):
    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    initializer = tf.glorot_normal_initializer()

    with tf.variable_scope('cnn', reuse=reuse) as vs:
        cnn_inputs = \
            tf.reshape(inputs,
                       shape=[-1, inputs.shape[2],
                              inputs.shape[3], inputs.shape[4]])

        frame_cnn = tl.conv2d(inputs=cnn_inputs,
                              filters=64, kernel_size=(7, 7),
                              strides=(1, 1), padding='same',
                              data_format='channels_last',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='cnn_conv1')

        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            frame_cnn = tl.batch_normalization(inputs=frame_cnn,
                                               training=is_training,
                                               name='cnn_bn1')
        if use_layer_norm is True:
            frame_cnn = tf.contrib.layers.layer_norm(inputs=frame_cnn)

    # conv1_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                 (32 * 16,               64,       64,       32        )
        frame_cnn = tl.max_pooling2d(inputs=frame_cnn,
                                     pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='valid',
                                     name='cnn_pool1')
        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])
    # pool1_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                 (32         * 16,       32,       32,       32        )

        frame_cnn = tl.conv2d(inputs=frame_cnn,
                              filters=64, kernel_size=(3, 3),
                              strides=(1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='cnn_conv2')
        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            frame_cnn = tl.batch_normalization(inputs=frame_cnn,
                                               training=is_training,
                                               name='cnn_bn2')
        if use_layer_norm is True:
            frame_cnn = tf.contrib.layers.layer_norm(inputs=frame_cnn)
    # conv2_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                 (32         * 16,       32,       32,       32       )

        frame_cnn = tl.conv2d(inputs=frame_cnn,
                              filters=128, kernel_size=(3, 3),
                              strides=(1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='cnn_conv3')
        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            frame_cnn = tl.batch_normalization(inputs=frame_cnn,
                                               training=is_training,
                                               name='cnn_bn3')
        if use_layer_norm is True:
            frame_cnn = tf.contrib.layers.layer_norm(inputs=frame_cnn)
    # conv3a_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                  (32         * 16,        32,       32,       64      )

        frame_cnn = tl.max_pooling2d(inputs=frame_cnn,
                                     pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='valid',
                                     name='cnn_pool3')

        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])
    # pool3_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                 (32         * 8,        16,        16,        64       )
        frame_cnn = tl.conv2d(inputs=frame_cnn,
                              filters=128, kernel_size=(3, 3),
                              strides=(1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='cnn_conv4')
        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            frame_cnn = tl.batch_normalization(inputs=frame_cnn,
                                               training=is_training,
                                               name='cnn_bn4')
        if use_layer_norm is True:
            frame_cnn = tf.contrib.layers.layer_norm(inputs=frame_cnn)
    # conv4_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                  (32        * 8,        16,        16,       64       )

        frame_cnn = tl.max_pooling2d(inputs=frame_cnn,
                                     pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='valid',
                                     name='cnn_pool4')
        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])

    # pool4_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                 (32         * 4,        8,        8,           64       )
        frame_cnn = tl.conv2d(inputs=frame_cnn,
                              filters=256, kernel_size=(3, 3),
                              strides=(1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='cnn_conv5')
        if video_emb_layer_name == frame_cnn.name.split('/')[1]:
            video_emb = tl.flatten(frame_cnn)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            frame_cnn = tl.batch_normalization(inputs=frame_cnn,
                                               training=is_training,
                                               name='cnn_bn5')
        if use_layer_norm is True:
            frame_cnn = tf.contrib.layers.layer_norm(inputs=frame_cnn)
    # conv5_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                  (32        * 4,        8,        8,          128       )

        cnn_feats = tl.max_pooling2d(inputs=frame_cnn,
                                     pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding='valid',
                                     name='cnn_pool5')
        if video_emb_layer_name == cnn_feats.name.split('/')[1]:
            video_emb = tl.flatten(cnn_feats)
            emb_dim = int(video_emb.shape[1])

    # pool5_output -> (batch_size * n_frames, img_size, img_size, n_channels)
    #                 (32         * 2,        4,        4,        128       )

        cnn_feats = tl.flatten(cnn_feats)
        cnn_emb_dim = int(cnn_feats.shape[1])

        if is_multilabel is True:
            cnn_softmax_logits = tl.dense(inputs=cnn_feats,
                                          units=2*n_classes,
                                          activation=None,
                                          kernel_regularizer=regularizer,
                                          name='cnn_softmax_logits')
            cnn_softmax_logits = tf.reshape(cnn_softmax_logits,
                                            (-1, n_classes, 2))
        else:
            cnn_softmax_logits = tl.dense(inputs=cnn_feats,
                                          units=n_classes,
                                          activation=None,
                                          kernel_regularizer=regularizer,
                                          name='cnn_softmax_logits')

        if is_training is False:
            cnn_softmax_logits = \
                tf.strided_slice(
                    cnn_softmax_logits,
                    [0, 0, 0], tf.shape(cnn_softmax_logits), [16, 1, 1],
                    name='cnn_softmax_logits_last')

        if cnn_softmax_logits.name.find(video_emb_layer_name) != -1:
            video_emb = tl.flatten(cnn_softmax_logits)
            emb_dim = int(video_emb.shape[1])

        cnn_softmax_out = tf.nn.sigmoid(cnn_softmax_logits)

    cnn_vars = tf.contrib.framework.get_variables(vs)

    with tf.variable_scope('lstm', reuse=reuse) as vs2:
        lstm_inputs = \
            tf.reshape(cnn_feats,
                       shape=[-1, inputs.shape[1],
                              cnn_feats.shape[1]])

        def lstm_cell():
            cell = tf.nn.rnn_cell.LSTMCell(
                num_units=512,
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
                name="lstm_cell")
            return cell

        stacked_lstm_cells = \
            tf.nn.rnn_cell.MultiRNNCell(
                cells=[lstm_cell() for _ in range(5)],
                state_is_tuple=True)

        outputs, state = \
            tf.nn.dynamic_rnn(
                cell=stacked_lstm_cells,
                inputs=lstm_inputs,
                dtype=tf.float32)

        if is_training is True:
            lstm_feats = tf.reshape(outputs, [-1, outputs.shape[-1]],
                                    name='lstm_out')
        else:
            last_indice = outputs.get_shape().as_list()[1] - 1
            lstm_feats = \
                tf.slice(outputs, [0, last_indice, 0], [-1, -1, -1],
                         name='lstm_out')

        if lstm_feats.name.split('/')[1].find(video_emb_layer_name) != -1:
            video_emb = tl.flatten(lstm_feats)
            emb_dim = int(video_emb.shape[1])

        if is_multilabel is True:
            softmax_logits = tl.dense(inputs=lstm_feats,
                                      units=2*n_classes,
                                      activation=None,
                                      kernel_regularizer=regularizer,
                                      name='softmax_logits')
            softmax_logits = tf.reshape(softmax_logits,
                                        (-1, n_classes, 2))
        else:
            softmax_logits = tl.dense(inputs=frame_cnn,
                                      units=n_classes,
                                      activation=None,
                                      kernel_regularizer=regularizer,
                                      name='softmax_logits')

        if video_emb_layer_name == softmax_logits.name.split('/')[1]:
            video_emb = tl.flatten(softmax_logits)
            emb_dim = int(video_emb.shape[1])

        softmax_out = tf.nn.sigmoid(softmax_logits)

    lstm_vars = tf.contrib.framework.get_variables(vs2)

    return (cnn_softmax_out, softmax_out), \
        (cnn_softmax_logits, softmax_logits), \
        (cnn_feats, video_emb), (cnn_vars, lstm_vars), \
        (cnn_emb_dim, emb_dim)
