import tensorflow as tf
tl = tf.layers


def conv_S(inputs, filters, kernel_size=(3, 3), strides=(1, 1),
           padding='valid', data_format='channels_last',
           dilation_rate=(1, 1), activation=None, use_bias=True,
           kernel_initializer=None,
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None, kernel_constraint=None,
           bias_constraint=None, trainable=True, name=None, reuse=None):
    return tl.conv3d(inputs=inputs,
                     filters=filters,
                     kernel_size=(1,) + kernel_size,
                     strides=(1,) + strides,
                     padding=padding,
                     data_format=data_format,
                     dilation_rate=(1,) + dilation_rate,
                     activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     trainable=trainable,
                     name=name,
                     reuse=reuse)


def conv_T(inputs, filters, kernel_size=(3,), strides=(1,),
           padding='valid', data_format='channels_last',
           dilation_rate=(1,), activation=None, use_bias=True,
           kernel_initializer=None,
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=None, bias_regularizer=None,
           activity_regularizer=None, kernel_constraint=None,
           bias_constraint=None, trainable=True, name=None, reuse=None):
    return tl.conv3d(inputs=inputs,
                     filters=filters,
                     kernel_size=kernel_size + (1, 1),
                     strides=strides + (1, 1),
                     padding=padding,
                     data_format=data_format,
                     dilation_rate=dilation_rate + (1, 1),
                     activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     trainable=trainable,
                     name=name,
                     reuse=reuse)


def conv_DS(inputs, bottleneck_size,
            padding='valid', data_format='channels_last',
            activation=None, use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None, trainable=True, name=None, reuse=None):
    return tl.conv3d(inputs=inputs,
                     filters=bottleneck_size,
                     kernel_size=(1, 1, 1),
                     strides=(1, 1, 1),
                     padding=padding,
                     data_format=data_format,
                     dilation_rate=(1, 1, 1),
                     activation=activation,
                     use_bias=use_bias,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     activity_regularizer=activity_regularizer,
                     kernel_constraint=kernel_constraint,
                     bias_constraint=bias_constraint,
                     trainable=trainable,
                     name=name,
                     reuse=reuse)


def p3d_A(inputs, bottleneck_size, s_filters, t_filters, out_filters,
          training=True, reuse=False):
    with tf.variable_scope("P3D-A", reuse=reuse):
        # Bottleneck convolution
        p3d = conv_DS(inputs=inputs, bottleneck_size=bottleneck_size,
                      padding='valid', activation=None, use_bias=True,
                      name='conv_btn')
        p3d = tl.batch_normalization(inputs=p3d, momentum=0.99,
                                     epsilon=0.001, training=training,
                                     name='bn1',
                                     renorm=False, renorm_clipping=None,
                                     renorm_momentum=0.99, fused=None)
        # p3d = tf.nn.relu(p3d)

        # Spatial convolution
        p3d = conv_S(inputs=p3d, filters=s_filters, strides=(1, 1),
                     padding='valid', activation=None, use_bias=True,
                     name='conv_S')
        p3d = tl.batch_normalization(inputs=p3d, momentum=0.99,
                                     epsilon=0.001, training=training,
                                     name='bn2',
                                     renorm=False, renorm_clipping=None,
                                     renorm_momentum=0.99, fused=None)
        p3d = tf.nn.relu(p3d)

        # Temporal convolution
        p3d = conv_T(inputs=p3d, filters=t_filters, strides=(1,),
                     padding='valid', activation=None, name='conv_T')
        p3d = tl.batch_normalization(inputs=p3d, momentum=0.99,
                                     epsilon=0.001, training=training,
                                     name='bn3',
                                     renorm=False, renorm_clipping=None,
                                     renorm_momentum=0.99, fused=None)
        p3d = tf.nn.relu(p3d)

        # Reverse bottleneck convolution
        p3d = conv_DS(inputs=p3d, bottleneck_size=out_filters,
                      padding='valid', activation=None, use_bias=True,
                      name='conv_rbtn')
        p3d = tl.batch_normalization(inputs=p3d, momentum=0.99,
                                     epsilon=0.001, training=training,
                                     name='bn4',
                                     renorm=False, renorm_clipping=None,
                                     renorm_momentum=0.99, fused=None)
    return tf.nn.relu(inputs + p3d)


def p3d_B(inputs, bottleneck_size, s_filters, t_filters, out_filters,
          training=True, reuse=False):
    with tf.variable_scope("P3D-B", reuse=reuse):
        p3d_btn = conv_DS(inputs=inputs, bottleneck_size=bottleneck_size,
                          padding='valid', activation=None,
                          use_bias=True, name='conv_btn')
        p3d_btn = tl.batch_normalization(inputs=p3d_btn, momentum=0.99,
                                         epsilon=0.001, training=training,
                                         name='bn1',
                                         renorm=False, renorm_clipping=None,
                                         renorm_momentum=0.99, fused=None)
        p3d_btn = tf.nn.relu(p3d_btn)

        p3d_s = conv_S(inputs=p3d_btn, filters=s_filters, strides=(1, 1),
                       padding='valid', activation=None,
                       use_bias=True, name='conv_S')
        p3d_s = tl.batch_normalization(inputs=p3d_s, momentum=0.99,
                                       epsilon=0.001, training=training,
                                       name='bn2',
                                       renorm=False, renorm_clipping=None,
                                       renorm_momentum=0.99, fused=None)
        p3d_s = tf.nn.relu(p3d_s)

        p3d_t = conv_T(inputs=p3d_btn, filters=t_filters, strides=(1,),
                       padding='valid', activation=None, name='conv_T')
        p3d_t = tl.batch_normalization(inputs=p3d_t, momentum=0.99,
                                       epsilon=0.001, training=training,
                                       name='bn3',
                                       renorm=False, renorm_clipping=None,
                                       renorm_momentum=0.99, fused=None)
        # p3d_t = tf.nn.relu(p3d_t)

        p3d = conv_DS(inputs=tf.nn.relu(p3d_s + p3d_t),
                      bottleneck_size=out_filters,
                      padding='valid', activation=None, use_bias=True,
                      name='conv_rbtn')
    return tf.nn.relu(inputs + p3d)


def p3d_C(inputs, bottleneck_size, s_filters, t_filters, training=True,
          reuse=False):
    with tf.variable_scope("P3D-C"):
        p3d_btn = conv_DS(inputs=inputs, bottleneck_size=bottleneck_size,
                          padding='valid', activation=tf.nn.relu,
                          use_bias=True, name='conv_btn')
        # batch_normalization
        # relu

        p3d_s = conv_S(inputs=p3d_btn, filters=s_filters, strides=(1, 1),
                       padding='valid', activation=tf.nn.relu,
                       use_bias=True, name='conv_S')
        # batch_normalization
        # relu

        p3d_t = conv_T(inputs=p3d_s, filters=t_filters, strides=(1,),
                       padding='valid', activation=None, name='conv_T')
        # batch_normalization
        # relu

        p3d = conv_DS(inputs=tf.nn.relu(p3d_s + p3d_t),
                      bottleneck_size=inputs.shape[-1],
                      padding='valid', activation=None, use_bias=True,
                      name='conv_rbtn')
    return tf.nn.relu(inputs + p3d)


def r21d(inputs, s_filters, t_filters, training=True, first_block=False
         reuse=False):
    with tf.variable_scope("R(2+1)D", reuse=reuse):
        # Spatial convolution
        r21d = conv_S(inputs=inputs, filters=s_filters, strides=(2, 2),
                      padding='valid', activation=None, use_bias=True,
                      name='conv1_S')
        r21d = tl.batch_normalization(inputs=r21d, axis=3, momentum=0.9,
                                      epsilon=1e-3, center=True,
                                      scale=True, training=training,
                                      fused=True, name='bn1_S')
        p3d = tf.nn.relu(p3d)

        # Temporal convolution
        p3d = conv_T(inputs=p3d, filters=t_filters, strides=(1,),
                     padding='valid', activation=None, name='conv_T')
        p3d = tl.batch_normalization(inputs=p3d, momentum=0.99,
                                     epsilon=0.001, training=training,
                                     name='bn3',
                                     renorm=False, renorm_clipping=None,
                                     renorm_momentum=0.99, fused=None)
        p3d = tf.nn.relu(p3d)

        # Reverse bottleneck convolution
        p3d = conv_DS(inputs=p3d, bottleneck_size=out_filters,
                      padding='valid', activation=None, use_bias=True,
                      name='conv_rbtn')
        p3d = tl.batch_normalization(inputs=p3d, momentum=0.99,
                                     epsilon=0.001, training=training,
                                     name='bn4',
                                     renorm=False, renorm_clipping=None,
                                     renorm_momentum=0.99, fused=None)
        model.ConvNd(
            data,
            'conv1_middle',
            num_input_channels,
            45,
            [1, 7, 7],
            weight_init=("MSRAFill", {}),
            strides=[1, 2, 2],
            pads=[0, 3, 3] * 2,
            no_bias=no_bias
        )

        model.SpatialBN(
            'conv1_middle',
            'conv1_middle_spatbn_relu',
            45,
            epsilon=1e-3,
            momentum=spatial_bn_mom,
            is_test=is_test
        )
        model.Relu('conv1_middle_spatbn_relu', 'conv1_middle_spatbn_relu')

        model.ConvNd(
            'conv1_middle_spatbn_relu',
            'conv1',
            45,
            64,
            [3, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[1, 1, 1],
            pads=[1, 0, 0] * 2,
            no_bias=no_bias
        )

    raise NotImplementedError


def def_p3d_small_video_ae(input, is_training=True, reuse=False,
                           use_l2_reg=False,
                           use_batch_norm=False, use_layer_norm=False,
                           video_ae_emb_layer_name='video_ae_emb_layer',
                           emb_dim=1024):
    raise NotImplementedError


def def_p3d_large_video_ae(
        input, is_training=True, reuse=False,
        use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_ae_emb_layer_name='clf_fc6'):
    raise NotImplementedError


def def_p3d_small_video_classifier(
        input, is_training=True, reuse=False,
        use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_ae_emb_layer_name='clf_fc6'):
    raise NotImplementedError


def def_p3d_large_video_classifer(
        input, is_training=True, reuse=False,
        use_l2_reg=False,
        use_batch_norm=False,
        use_layer_norm=False,
        video_ae_emb_layer_name='video_ae_emb_layer',
        emb_dim=1024):
    raise NotImplementedError


# Based on: https://git.io/fpwJK
def def_residual_block(inputs, filters, strides,
                       training, projection_shortcut, name=None):
    shortcut = inputs
    inputs = tl.batch_normalization(
        inputs=inputs, axis=3, momentum=0.997, epsilon=1e-5, center=True,
        scale=True, training=training, fused=True, name=name)

    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs =
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut


def def_r3d_small_video_classifier(
        input, is_training=True, reuse=False,
        use_l2_reg=False,
        use_batch_norm=False, use_layer_norm=False, use_dropout=True,
        video_ae_emb_layer_name='clf_fc6'):

    if use_l2_reg:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    else:
        regularizer = None

    with tf.variable_scope('video_classifier', reuse=reuse) as vs:
        video_clf = tl.conv3d(inputs=input,
                              filters=64, kernel_size=(3, 7, 7),
                              strides=(1, 2, 2), padding='same',
                              data_format='channels_last',
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name='clf_conv1')

        if video_ae_emb_layer_name == video_clf.name.split('/')[1]:
            video_emb = tl.flatten(video_clf)
            emb_dim = int(video_emb.shape[1])

        if use_batch_norm is True:
            video_clf = tl.batch_normalization(inputs=video_clf,
                                               training=is_training,
                                               name='clf_bn1')
        if use_layer_norm is True:
            video_clf = tf.contrib.layers.layer_norm(inputs=video_clf)

        video_clf = def_residual_block(
            inputs=video_clf, filters=64, strides=(2, 2, 2),
            training=is_training, projection_shortcut=False,
            name='clf_res1')

        video_clf = def_residual_block(
            inputs=video_clf, filters=64, strides=(2, 2, 2),
            training=is_training, projection_shortcut=False,
            name='clf_res2')
