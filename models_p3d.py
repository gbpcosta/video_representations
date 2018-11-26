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
        # batch_normalization
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
        # batch_normalization
    return tf.nn.relu(inputs + p3d)


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


def def_p3d_large_video_classifer(input, is_training=True, reuse=False,
                           use_l2_reg=False,
                           use_batch_norm=False, use_layer_norm=False,
                           video_ae_emb_layer_name='video_ae_emb_layer',
                           emb_dim=1024):
    raise NotImplementedError
