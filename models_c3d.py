import tensorflow as tf
tl = tf.layers


def def_c3d_small_video_ae(input, is_training=True, reuse=False,
                           use_l2_reg=False,
                           use_batch_norm=False, use_layer_norm=False,
                           video_emb_layer_name='video_ae_emb_layer'):
    """ VIDEO AE """

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    with tf.variable_scope('video_ae_encoder', reuse=reuse) as vs:
        video_ae_en = tl.conv3d(inputs=input,
                                filters=32, kernel_size=(3, 3, 3),
                                strides=(2, 2, 2), padding='same',
                                data_format='channels_last',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv1')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn1')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,       32,       32,       32        )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(2, 2, 2),
                                       strides=(2, 2, 2),
                                       padding='valid',
                                       name='ven_pool1')
    # pool1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,       16,       16,       32        )

        video_ae_en = tl.conv3d(inputs=video_ae_en,
                                filters=64, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding='same',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv2')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn2')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,       16,       16,       64       )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(2, 2, 2),
                                       strides=(2, 2, 2),
                                       padding='valid',
                                       name='ven_pool2')
    # pool2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        8,       8,       64       )

        video_ae_en = tl.conv3d(inputs=video_ae_en,
                                filters=64, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding='same',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv3')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn3')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         2,        8,       8,       64       )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(1, 2, 2),
                                       strides=(1, 2, 2),
                                       padding='valid',
                                       name='ven_pool3')
    # pool3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        4,        4,        64       )

        video_ae_en = tl.flatten(inputs=video_ae_en,
                                 name='ven_flatten')
    # flatten_output -> (batch_size, features)
    #                   (32,         2048    )

    with tf.variable_scope(video_emb_layer_name, reuse=reuse) as vs2:
        """ CODE """
        video_ae_emb = tl.dense(inputs=video_ae_en,
                                units=1024,
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='emb_space')
        emb_dim = 1024

    with tf.variable_scope('video_ae_decoder', reuse=reuse) as vs3:
        """ DECODER """
    # Input -> (batch_size, code_size   )
    #          (32,         self.emb_dim)
        video_ae_de = tl.dense(inputs=video_ae_emb,
                               units=2048, activation=tf.nn.relu,
                               kernel_regularizer=regularizer,
                               name='vde_fc1')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn1')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)

        video_ae_de = tf.reshape(tensor=video_ae_de,
                                 shape=(-1, 2, 4, 4, 64),
                                 name='vde_reshape')
    # reshape_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                   (32,         2,        4,        4,        64      )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=64, kernel_size=(3, 3, 3),
            strides=(2, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp1')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn2')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,        8,        8,        64       )
    #         video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                    name='de_up1')(video_ae_de)
    # up1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         4,        8,        8,        64       )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=64, kernel_size=(3, 3, 3),
            strides=(2, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp2')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn3')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,        16,        16,        64      )
    #     video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                name='de_up2')(video_ae_de)
    # up2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         8,        16,       16,       64       )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=32, kernel_size=(3, 3, 3),
            strides=(2, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp3')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn4')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,        32,       32,       32       )
    #     video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                name='de_up3')(video_ae_de)
    # up3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         16,       32,       32,       32        )
        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=32, kernel_size=(3, 3, 3),
            strides=(1, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp4')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn5')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,        64,       64,       32       )
    #     video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                name='de_up3')(video_ae_de)
    # up4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         16,       64,       64,       32        )

        video_ae_recon_logits = tl.conv3d(inputs=video_ae_de,
                                          filters=1,
                                          kernel_size=(3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same',
                                          activation=None,
                                          kernel_regularizer=regularizer,
                                          name='video_ae_recon_logits')
    # conv5_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       64,       64,       1         )
        video_ae_recon = tf.nn.sigmoid(video_ae_recon_logits,
                                       name='video_ae_recon')

    video_ae_vars = tf.contrib.framework.get_variables(vs) + \
        tf.contrib.framework.get_variables(vs2) + \
        tf.contrib.framework.get_variables(vs3)

    return video_ae_recon, video_ae_recon_logits, \
        video_ae_emb, video_ae_vars, emb_dim


def def_c3d_large_video_ae(input, is_training=True, reuse=False,
                           use_l2_reg=False,
                           use_batch_norm=False, use_layer_norm=False,
                           use_dropout=True,
                           video_emb_layer_name='video_ae_emb_layer'):
    """ VIDEO AE """

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    with tf.variable_scope('video_ae_encoder', reuse=reuse) as vs:
        video_ae_en = tl.conv3d(inputs=input,
                                filters=64, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding='same',
                                data_format='channels_last',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv1')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn1')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       64,       64,       32        )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(1, 2, 2),
                                       strides=(1, 2, 2),
                                       padding='valid',
                                       name='ven_pool1')
    # pool1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32        )

        video_ae_en = tl.conv3d(inputs=video_ae_en,
                                filters=128, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding='same',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv2')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn2')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32       )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(2, 2, 2),
                                       strides=(2, 2, 2),
                                       padding='valid',
                                       name='ven_pool2')
    # pool2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,        16,       16,       32       )

        video_ae_en = tl.conv3d(inputs=video_ae_en,
                                filters=128, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding='same',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv3')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn3')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         8,        16,       16,       64       )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(2, 2, 2),
                                       strides=(2, 2, 2),
                                       padding='valid',
                                       name='ven_pool3')
    # pool3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,        8,        8,        64       )
        video_ae_en = tl.conv3d(inputs=video_ae_en,
                                filters=64, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1), padding='same',
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='ven_conv4')
        if use_batch_norm is True:
            video_ae_en = tl.batch_normalization(inputs=video_ae_en,
                                                 training=is_training,
                                                 name='ven_bn4')
        if use_layer_norm is True:
            video_ae_en = tf.contrib.layers.layer_norm(inputs=video_ae_en)
    # conv4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         4,        8,        8,       64       )
        video_ae_en = tl.max_pooling3d(inputs=video_ae_en,
                                       pool_size=(2, 2, 2),
                                       strides=(2, 2, 2),
                                       padding='valid',
                                       name='ven_pool4')
    # pool4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        4,        4,        64       )

        video_ae_en = tl.flatten(inputs=video_ae_en,
                                 name='ven_flatten')
    # flatten_output -> (batch_size, features)
    #                   (32,         2048    )

    with tf.variable_scope(video_emb_layer_name, reuse=reuse) as vs2:
        """ CODE """
        video_ae_emb = tl.dense(inputs=video_ae_en,
                                units=2048,
                                activation=tf.nn.relu,
                                kernel_regularizer=regularizer,
                                name='emb_space')
        emb_dim = 2048

    with tf.variable_scope('video_ae_decoder', reuse=reuse) as vs3:
        """ DECODER """
    # Input -> (batch_size, code_size   )
    #          (32,         self.emb_dim)
        video_ae_de = tl.dense(inputs=video_ae_emb,
                               units=2048, activation=tf.nn.relu,
                               kernel_regularizer=regularizer,
                               name='vde_fc1')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn1')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)

        video_ae_de = tf.reshape(tensor=video_ae_de,
                                 shape=(-1, 2, 4, 4, 64),
                                 name='vde_reshape')
    # reshape_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                   (32,         2,        4,        4,        128       )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=128, kernel_size=(3, 3, 3),
            strides=(2, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp1')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn2')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        4,        4,        128       )
    #         video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                    name='de_up1')(video_ae_de)
    # up1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         4,        8,        8,        128       )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=128, kernel_size=(3, 3, 3),
            strides=(2, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp2')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn3')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,        8,        8,        128       )
    #     video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                name='de_up2')(video_ae_de)
    # up2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         8,        16,       16,       128       )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=64, kernel_size=(3, 3, 3),
            strides=(2, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp3')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn4')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,        16,       16,       64        )
    #     video_ae_de = UpSampling3D(size=(2, 2, 2),
    #                                name='de_up3')(video_ae_de)
    # up3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         16,       32,       32,       64        )

        video_ae_de = tl.conv3d_transpose(
            inputs=video_ae_de,
            filters=32, kernel_size=(3, 3, 3),
            strides=(1, 2, 2), padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name='vde_conv_transp4')
        if use_batch_norm is True:
            video_ae_de = tl.batch_normalization(inputs=video_ae_de,
                                                 training=is_training,
                                                 name='vde_bn5')
        if use_layer_norm is True:
            video_ae_de = tf.contrib.layers.layer_norm(inputs=video_ae_de)
    # conv4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32        )
    #     video_ae_de = UpSampling3D(size=(1, 2, 2),
    #                                name='de_up4')(video_ae_de)
    # up4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #               (32,         16,       64,       64,       32        )

        video_ae_recon_logits = tl.conv3d(inputs=video_ae_de,
                                          filters=1,
                                          kernel_size=(3, 3, 3),
                                          strides=(1, 1, 1),
                                          padding='same',
                                          activation=None,
                                          kernel_regularizer=regularizer,
                                          name='video_ae_recon_logits')
    # conv4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       1         )
        video_ae_recon = tf.nn.sigmoid(video_ae_recon_logits,
                                       name='video_ae_recon')

    video_ae_vars = tf.contrib.framework.get_variables(vs) + \
        tf.contrib.framework.get_variables(vs2) + \
        tf.contrib.framework.get_variables(vs3)

    return video_ae_recon, video_ae_recon_logits, \
        video_ae_emb, video_ae_vars, emb_dim


def def_c3d_small_video_classifier_old(
        input, n_classes, is_training=True,
        is_multilabel=False, reuse=False, use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_emb_layer_name='clf_fc5'):

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    with tf.variable_scope('video_classifier', reuse=reuse) as vs:
        video_clf = tl.conv3d(inputs=input,
                              filters=64, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              data_format='channels_last',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv1')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn1')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       64,       64,       32        )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(1, 2, 2),
                                     strides=(1, 2, 2),
                                     padding='valid',
                                     name='clf_pool1')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])
    # pool1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32        )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=64, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv2')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn2')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32       )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool2')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,        16,       16,       32       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=128, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv3')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn3')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv3a_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         8,        16,       16,       128      )

        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool3')

        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])
    # pool3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,        8,        8,        128       )
        video_clf = tl.conv3d(inputs=video_clf,
                              filters=128, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv4')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn4a')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv4a_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         4,        8,        8,       512       )

        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool4')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        4,        4,        128       )

        video_clf = tl.flatten(inputs=video_clf,
                               name='clf_flatten')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # flatten_output -> (batch_size, features)
    #                   (32,         4096    )

        video_clf = tl.dense(inputs=video_clf,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_regularizer=regularizer,
                             name='clf_fc5')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_dropout is True:
            video_clf = tl.dropout(inputs=video_clf,
                                   rate=0.5,
                                   training=is_training,
                                   name=None)

        if is_multilabel is True:
            video_clf_out_logits = tl.dense(inputs=video_clf,
                                            units=2*n_classes,
                                            activation=None,
                                            kernel_regularizer=regularizer,
                                            name='clf_fc6')
            video_clf_out_logits = tf.reshape(video_clf_out_logits,
                                              (-1, n_classes, 2))
        else:
            video_clf_out_logits = tl.dense(inputs=video_clf,
                                            units=n_classes,
                                            activation=None,
                                            kernel_regularizer=regularizer,
                                            name='clf_fc6')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf_out_logits)
            emb_dim = int(video_emb.shape[1])

        video_clf_out = tf.nn.sigmoid(video_clf_out_logits)

    video_clf_vars = tf.contrib.framework.get_variables(vs)

    return video_clf_out, video_clf_out_logits, \
        video_emb, video_clf_vars, emb_dim


def def_c3d_large_video_classifier(
        input, n_classes, is_training=True,
        is_multilabel=False, reuse=False, use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_emb_layer_name='clf_fc6'):

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    with tf.variable_scope('video_classifier', reuse=reuse) as vs:
        video_clf = tl.conv3d(inputs=input,
                              filters=64, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              data_format='channels_last',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv1')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn1')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       64,       64,       32        )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(1, 2, 2),
                                     strides=(1, 2, 2),
                                     padding='valid',
                                     name='clf_pool1')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])
    # pool1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32        )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=128, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv2')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn2')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32       )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool2')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,        16,       16,       32       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=256, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv3a')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn3a')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv3a_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         8,        16,       16,       256       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=256, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv3b')

        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn3b')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv3b_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         8,        16,       16,       256       )

        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool3')

        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])
    # pool3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,        8,        8,        256       )
        video_clf = tl.conv3d(inputs=video_clf,
                              filters=512, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv4a')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn4a')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv4a_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         4,        8,        8,       512       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=512, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv4b')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn4b')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv4b_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         4,        8,        8,       512       )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool4')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        4,        4,        512       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=512, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv5a')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn5a')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv5a_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         2,        4,        4,       512       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=512, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv5b')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn5b')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv5b_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         2,        4,        4,       512       )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool5')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool5_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         1,        2,        2,        512       )

        video_clf = tl.flatten(inputs=video_clf,
                               name='clf_flatten')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # flatten_output -> (batch_size, features)
    #                   (32,         2048    )

        video_clf = tl.dense(inputs=video_clf,
                             units=4096,
                             activation=tf.nn.relu,
                             kernel_regularizer=regularizer,
                             name='clf_fc6')

        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_dropout is True:
            video_clf = tl.dropout(inputs=video_clf,
                                   rate=0.5,
                                   training=is_training,
                                   name=None)

        video_clf = tl.dense(inputs=video_clf,
                             units=2048,
                             activation=tf.nn.relu,
                             kernel_regularizer=regularizer,
                             name='clf_fc7')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_dropout is True:
            video_clf = tl.dropout(inputs=video_clf,
                                   rate=0.5,
                                   training=is_training,
                                   name=None)

        if is_multilabel is True:
            video_clf_out_logits = tl.dense(inputs=video_clf,
                                            units=2*n_classes,
                                            activation=None,
                                            kernel_regularizer=regularizer,
                                            name='clf_fc8')
            video_clf_out_logits = tf.reshape(video_clf_out_logits,
                                              (-1, n_classes, 2))
        else:
            video_clf_out_logits = tl.dense(inputs=video_clf,
                                            units=n_classes,
                                            activation=None,
                                            kernel_regularizer=regularizer,
                                            name='clf_fc8')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf_out_logits)
            emb_dim = int(video_emb.shape[1])

        video_clf_out = tf.nn.sigmoid(video_clf_out_logits)

    video_clf_vars = tf.contrib.framework.get_variables(vs)

    return video_clf_out, video_clf_out_logits, \
        video_emb, video_clf_vars, emb_dim


def def_c3d_small_video_classifier(
        input, n_classes, is_training=True,
        is_multilabel=False, reuse=False, use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_emb_layer_name='clf_flatten'):

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    with tf.variable_scope('video_classifier', reuse=reuse) as vs:
        video_clf = tl.conv3d(inputs=input,
                              filters=32, kernel_size=(3, 7, 7),
                              strides=(1, 1, 1), padding='same',
                              data_format='channels_last',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv1')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn1')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)

        # video_clf = tf.nn.relu(video_clf)
    # conv1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       64,       64,       32        )
        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(1, 2, 2),
                                     strides=(1, 2, 2),
                                     padding='valid',
                                     name='clf_pool1')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])
    # pool1_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32        )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=32, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv2')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn2')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv2_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         16,       32,       32,       32       )

        video_clf = tl.conv3d(inputs=video_clf,
                              filters=64, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv3')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn3')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv3a_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         16,        32,       32,       64      )

        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool3')

        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])
    # pool3_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         8,        16,        16,        64       )
        video_clf = tl.conv3d(inputs=video_clf,
                              filters=64, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv4')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn4')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         8,        16,        16,       64       )

        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool4')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool4_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         4,        8,        8,        64       )
        video_clf = tl.conv3d(inputs=video_clf,
                              filters=128, kernel_size=(3, 3, 3),
                              strides=(1, 1, 1), padding='same',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv5')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn5')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)
    # conv5_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                  (32,         4,        8,        8,       128       )

        video_clf = tl.max_pooling3d(inputs=video_clf,
                                     pool_size=(2, 2, 2),
                                     strides=(2, 2, 2),
                                     padding='valid',
                                     name='clf_pool5')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # pool5_output -> (batch_size, n_frames, img_size, img_size, n_channels)
    #                 (32,         2,        4,        4,        128       )


        video_clf = tl.flatten(inputs=video_clf,
                               name='clf_flatten')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

    # flatten_output -> (batch_size, features)
    #                   (32,         4096    )

        if is_multilabel is True:
            video_clf_out_logits = tl.dense(inputs=video_clf,
                                            units=2*n_classes,
                                            activation=None,
                                            kernel_regularizer=regularizer,
                                            name='clf_fc6')
            video_clf_out_logits = tf.reshape(video_clf_out_logits,
                                              (-1, n_classes, 2))
        else:
            video_clf_out_logits = tl.dense(inputs=video_clf,
                                            units=n_classes,
                                            activation=None,
                                            kernel_regularizer=regularizer,
                                            name='clf_fc6')
        if video_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf_out_logits)
            emb_dim = int(video_emb.shape[1])

        video_clf_out = tf.nn.sigmoid(video_clf_out_logits)

    video_clf_vars = tf.contrib.framework.get_variables(vs)

    return video_clf_out, video_clf_out_logits, \
        video_emb, video_clf_vars, emb_dim
