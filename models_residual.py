import tensorflow as tf
tl = tf.layers


# Based on: https://git.io/fpwt6

BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 3, 4, 3),
    34: (3, 4, 6, 3),
}


class VideoModelBuilder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(self, input, is_training=True, reuse=False,
                 verbosity=0):
        # model, prev_blob, no_bias,
        # self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.reuse = reuse
        # self.prev_blob = input
        self.is_training = is_training
        # self.spatial_bn_mom = spatial_bn_mom
        self.verbosity = verbosity
        # self.no_bias = 1 if no_bias else 0

    def add_conv(self, input, filters,
                 is_decomposed=False,  # set this to be True for (2+1)D conv
                 kernel_size=(3, 3, 3), strides=(1, 1, 1),
                 padding='same', reuse=False):
        self.comp_idx += 1

        with tf.variable_scope('conv%d' % (self.comp_count), reuse=reuse):
            if is_decomposed:
                i = 3 * int(input.shape[-1]) * filters * kernel_size[1] * \
                    kernel_size[2]
                i /= int(input.shape[-1]) * kernel_size[1] * kernel_size[2] + \
                    3 * filters

                middle_filters = int(i)

                if self.verbosity >= 2:
                    print("Number of middle filters: "
                          "{}".format(middle_filters))

                prev_blob = tl.conv3d(
                    inputs=input,
                    filters=middle_filters,
                    kernel_size=(1, kernel_size[1], kernel_size[2]),
                    strides=(1, strides[1], strides[2]),
                    padding=padding,
                    activation=None,
                    # use_bias=use_bias,
                    # activity_regularizer=activity_regularizer,
                    name='S_%d_middle' % (self.comp_idx),
                    reuse=reuse)

                prev_blob = self.add_spatial_bn(prev_blob, suffix='_middle')
                prev_blob = self.add_relu(prev_blob)

                prev_blob = tl.conv3d(
                    inputs=prev_blob,
                    filters=filters,
                    kernel_size=(kernel_size[0], 1, 1),
                    strides=(strides[0], 1, 1),
                    padding=padding,
                    activation=None,
                    # use_bias=use_bias,
                    # activity_regularizer=activity_regularizer,
                    name='T_%d' % (self.comp_idx),
                    reuse=reuse)
            else:
                prev_blob = tl.conv3d(
                    inputs=input,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation=None,
                    # use_bias=use_bias,
                    # activity_regularizer=activity_regularizer,
                    name='ST_%d' % (self.comp_idx),
                    reuse=reuse)

            return prev_blob

    def add_relu(self, input):
        return tf.nn.relu(input)

    def add_spatial_bn(self, input, suffix=''):
        return tl.batch_normalization(inputs=input, axis=-1,
                                      momentum=0.9, epsilon=1e-3,
                                      center=True, scale=True,
                                      training=self.is_training,
                                      fused=True,
                                      name='S_bn%d%s' % (self.comp_idx,
                                                         suffix))

    '''
    Add a "bottleneck" component which can be 2d, 3d, (2+1)d
    '''
    def add_bottleneck(self, input,
                       base_filters,  # filters internally in the component
                       output_filters,  # feature maps to output
                       down_sampling=False, spatial_batch_norm=True,
                       is_decomposed=False, is_real_3d=True,
                       padding='same'):
        with tf.variable_scope('bottleneck_res%d' % (self.comp_count),
                               reuse=self.reuse):
            if is_decomposed:
                # decomposition can only be applied to 3d conv
                assert is_real_3d

            input_filters = int(input.shape[-1])

            self.comp_idx = 0
            shortcut_blob = input

            # 1x1x1
            prev_blob = self.add_conv(input=input, filters=base_filters,
                                      kernel_size=(1, 1, 1))

            if spatial_batch_norm:
                prev_blob = self.add_spatial_bn(input=prev_blob,
                                                is_training=is_training)
            prev_blob = self.add_relu(input=prev_blob)

            if down_sampling:
                if is_real_3d:
                    use_striding = (2, 2, 2)
                else:
                    use_striding = (1, 2, 2)
            else:
                use_striding = (1, 1, 1)

            # 3x3x3 (note the pad, required for keeping dimensions)
            prev_blob = self.add_conv(
                input=prev_blob,
                filters=base_filters,
                kernel_size=(3, 3, 3) if is_real_3d else (1, 3, 3),
                strides=use_striding,
                is_decomposed=is_decomposed)

            if spatial_batch_norm:
                prev_blob = self.add_spatial_bn(input=prev_blob)
            prev_blob = self.add_relu(input=prev_blob)

            # 1x1x1
            last_conv = self.add_conv(input=prev_blob,
                                      filters=output_filters,
                                      kernel_size=(1, 1, 1))
            if spatial_batch_norm:
                last_conv = self.add_spatial_bn(input=prev_blob)

            # Summation with input signal (shortcut)
            # If we need to increase dimensions (feature maps), need to
            # do do a projection for the short cut
            if (output_filters > input_filters):
                shortcut_blob = tl.conv3d(
                    inputs=shortcut_blob,
                    filters=output_filters,
                    kernel_size=(1, 1, 1),
                    strides=use_striding,
                    padding=padding,
                    activation=None,
                    # use_bias=use_bias,
                    # activity_regularizer=activity_regularizer,
                    name='shorcut_projection')

                if spatial_batch_norm:
                    shortcut_blob = tl.batch_normalization(
                        inputs=shortcut_blob, axis=-1,
                        momentum=0.9, epsilon=1e-3,
                        center=True, scale=True,
                        training=self.is_training,
                        fused=True,
                        name='shortcut_projection_bn')

            prev_blob = tf.add(shortcut_blob, last_conv,
                               name='sum_%d' % (self.comp_idx))
            self.comp_idx += 1
            prev_blob = self.add_relu(input=prev_blob)

            # Keep track of number of high level components
            self.comp_count += 1
            return prev_blob

    '''
    Add a "simple_block" component which can be 2d, 3d, (2+1)d
    '''
    def add_simple_block(self, input, filters,
                         down_sampling=False, spatial_batch_norm=True,
                         is_decomposed=False, is_real_3d=True,
                         padding='same', only_spatial_downsampling=False):
        with tf.variable_scope('res%d' % (self.comp_count),
                               reuse=self.reuse):
            if is_decomposed:
                # decomposition can only be applied to 3d conv
                assert is_real_3d

            self.comp_idx = 0
            shortcut_blob = input
            input_filters = int(input.shape[-1])

            if down_sampling:
                if is_real_3d:
                    if only_spatial_downsampling:
                        use_striding = (1, 2, 2)
                    else:
                        use_striding = (2, 2, 2)
                else:
                    use_striding = (1, 2, 2)
            else:
                use_striding = (1, 1, 1)

            # 3x3x3
            prev_blob = self.add_conv(
                input=input, filters=filters,
                kernel_size=(3, 3, 3) if is_real_3d else (1, 3, 3),
                strides=use_striding, padding=padding,
                is_decomposed=is_decomposed, reuse=self.reuse)

            if spatial_batch_norm:
                prev_blob = self.add_spatial_bn(input=prev_blob)
            prev_blob = self.add_relu(input=prev_blob)

            last_conv = self.add_conv(
                input=prev_blob, filters=filters,
                kernel_size=(3, 3, 3) if is_real_3d else (1, 3, 3),
                padding=padding, is_decomposed=is_decomposed)

            if spatial_batch_norm:
                last_conv = self.add_spatial_bn(input=last_conv)

            if (filters != input_filters) or down_sampling:
                shortcut_blob = tl.conv3d(
                    inputs=shortcut_blob,
                    filters=filters,
                    kernel_size=(1, 1, 1),
                    strides=use_striding,
                    padding=padding,
                    activation=None,
                    # use_bias=use_bias,
                    # activity_regularizer=activity_regularizer,
                    name='shorcut_projection')

                if spatial_batch_norm:
                    shortcut_blob = tl.batch_normalization(
                        inputs=shortcut_blob, axis=-1,
                        momentum=0.9, epsilon=1e-3,
                        center=True, scale=True,
                        training=self.is_training,
                        fused=True,
                        name='shortcut_projection_bn')

            prev_blob = tf.add(shortcut_blob, last_conv,
                               name='sum%d' % (self.comp_idx))
            self.comp_idx += 1
            prev_blob = self.add_relu(input=prev_blob)

            # Keep track of number of high level components
            self.comp_count += 1
            return prev_blob


# 3d or (2+1)d resnets, input 3 x t*8 x 112 x 112
# the final conv output is 512 * t * 7 * 7
def def_r3d(input, num_labels, is_training=True,
            final_spatial_kernel=7, final_temporal_kernel=1,
            model_depth=18, padding='same', is_decomposed=False,
            verbosity=0, reuse=False, video_emb_layer_name='res7'):

    with tf.variable_scope('video_classifier', reuse=reuse) as vs:
        # conv1 + maxpool
        if not is_decomposed:
            prev_blob = tl.conv3d(
                inputs=input,
                filters=64,
                kernel_size=(3, 7, 7),
                strides=(1, 2, 2),
                padding=padding,
                activation=None,
                # use_bias=use_bias,
                # activity_regularizer=activity_regularizer,
                name='conv1',
                reuse=reuse)
        else:
            prev_blob = tl.conv3d(
                inputs=input,
                filters=45,
                kernel_size=(1, 7, 7),
                strides=(1, 2, 2),
                padding=padding,
                activation=None,
                # use_bias=use_bias,
                # activity_regularizer=activity_regularizer,
                name='conv1_middle',
                reuse=reuse)

            prev_blob = tl.batch_normalization(
                inputs=prev_blob, axis=-1,
                momentum=0.9, epsilon=1e-3,
                center=True, scale=True,
                training=is_training,
                fused=True, reuse=reuse,
                name='conv1_middle_bn')
            prev_blob = tf.nn.relu(prev_blob)

            prev_blob = tl.conv3d(
                inputs=prev_blob,
                filters=64,
                kernel_size=(3, 1, 1),
                strides=(1, 1, 1),
                padding=padding,
                activation=None,
                # use_bias=use_bias,
                # activity_regularizer=activity_regularizer,
                name='conv1', reuse=reuse)

        prev_blob = tl.batch_normalization(
            inputs=prev_blob, axis=-1,
            momentum=0.9, epsilon=1e-3,
            center=True, scale=True,
            training=is_training,
            fused=True, reuse=reuse,
            name='conv1_bn')
        prev_blob = tf.nn.relu(prev_blob)

        if video_emb_layer_name.startswith(prev_blob.name.split('/')[-1]):
            video_emb = tl.flatten(prev_blob)
            emb_dim = int(video_emb.shape[1])

        (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

        # Residual blocks...
        builder = VideoModelBuilder(input, is_training=is_training,
                                    reuse=reuse, verbosity=verbosity)

        # conv_2x
        for ii in range(n1):
            prev_blob = builder.add_simple_block(
                input=prev_blob, filters=64,
                is_decomposed=is_decomposed, is_real_3d=True,
                padding=padding)
            # print('n1 %d: ' % ii, prev_blob.name.split('/'))

            if video_emb_layer_name == prev_blob.name.split('/')[-2]:
                video_emb = tl.flatten(prev_blob)
                emb_dim = int(video_emb.shape[1])

        # conv_3x
        prev_blob = builder.add_simple_block(
            input=prev_blob, filters=128, down_sampling=True,
            is_decomposed=is_decomposed, padding=padding)

        if video_emb_layer_name == prev_blob.name.split('/')[-2]:
            video_emb = tl.flatten(prev_blob)
            emb_dim = int(video_emb.shape[1])

        for ii in range(n2 - 1):
            prev_blob = builder.add_simple_block(
                input=prev_blob, filters=128,
                is_decomposed=is_decomposed,
                padding=padding)
            # print('n2 %d: ' % ii, prev_blob.name.split('/'))

            if video_emb_layer_name == prev_blob.name.split('/')[-2]:
                video_emb = tl.flatten(prev_blob)
                emb_dim = int(video_emb.shape[1])

        # conv_4x
        prev_blob = builder.add_simple_block(
            input=prev_blob, filters=256, down_sampling=True,
            is_decomposed=is_decomposed, padding=padding)

        if video_emb_layer_name == prev_blob.name.split('/')[-2]:
            video_emb = tl.flatten(prev_blob)
            emb_dim = int(video_emb.shape[1])

        for ii in range(n3 - 1):
            prev_blob = builder.add_simple_block(
                input=prev_blob, filters=256,
                is_decomposed=is_decomposed,
                padding=padding)
            # print('n3 %d: ' % ii, prev_blob.name.split('/'))

            if video_emb_layer_name == prev_blob.name.split('/')[-2]:
                video_emb = tl.flatten(prev_blob)
                emb_dim = int(video_emb.shape[1])

        # conv_5x
        prev_blob = builder.add_simple_block(
            input=prev_blob, filters=512, down_sampling=True,
            is_decomposed=is_decomposed, padding=padding)

        # print('n4: ', prev_blob.name.split('/')[-2], video_emb_layer_name)
        if video_emb_layer_name == prev_blob.name.split('/')[-2]:
            video_emb = tl.flatten(prev_blob)
            emb_dim = int(video_emb.shape[1])

        for ii in range(n4 - 1):
            prev_blob = builder.add_simple_block(
                input=prev_blob, filters=512,
                is_decomposed=is_decomposed,
                padding=padding)
            # print('n4 %d: ' % ii, prev_blob.name.split('/'))

            if video_emb_layer_name == prev_blob.name.split('/')[-2]:
                video_emb = tl.flatten(prev_blob)
                emb_dim = int(video_emb.shape[1])

        # Final layers
        final_avg = tl.average_pooling3d(inputs=prev_blob,
                                         pool_size=(final_temporal_kernel,
                                                    final_spatial_kernel,
                                                    final_spatial_kernel),
                                         strides=(1, 1, 1), padding='same',
                                         name='final_avg')
        # print('final %d: ' % ii, prev_blob.name.split('/'))
        if video_emb_layer_name == final_avg.name.split('/')[-1]:
            video_emb = tl.flatten(final_avg)
            emb_dim = int(video_emb.shape[1])

        final_avg = tl.flatten(inputs=final_avg,
                               name='flatten')

        video_clf_out_logits = tl.dense(inputs=final_avg, units=num_labels,
                                        activation=None, name='softmax',
                                        reuse=reuse)
        video_clf_out = tf.nn.sigmoid(video_clf_out_logits)

    video_clf_vars = tf.contrib.framework.get_variables(vs)

    return video_clf_out, video_clf_out_logits, \
        video_emb, video_clf_vars, emb_dim
