import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json

# Model architecture

def init_nerf_model(D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
    # what is input ch views? -- input channel number for viewing direction
    # cos positional encoding is also put on viewing direction as well

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch, input_ch_views, type(
        input_ch), type(input_ch_views), use_viewdirs)
    input_ch = int(input_ch)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch + input_ch_views))
    inputs_pts, inputs_views = tf.split(inputs, [input_ch, input_ch_views], -1)
    inputs_pts.set_shape([None, input_ch])
    inputs_views.set_shape([None, input_ch_views])

    print(inputs.shape, inputs_pts.shape, inputs_views.shape)
    outputs = inputs_pts
    for i in range(D):
        outputs = dense(W)(outputs)
        # what is dense(outputs)?
        # seems to be a dense layer that acts as a function that takes input and returns output
        # https://keras.io/guides/functional_api/
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        # alpha is the density of target point
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        # The supplement to the paper states there are 4 hidden layers here, but this is an error since
        # the experiments were actually run with 1 hidden layer, so we will leave it as 1.
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        # this outputs is r,g,b of target point
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def init_nerf_res_model(D=8, W=256, input_ch_image=(400, 400, 3), input_ch_coord=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
    # what is input ch views? -- input channel number for viewing direction
    # cos positional encoding is also put on viewing direction as well

    print("Initing nerf_res model")

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)

    print('MODEL', input_ch_coord, input_ch_views, type(
        input_ch_coord), type(input_ch_views), use_viewdirs)
    input_ch_coord = int(input_ch_coord)
    input_ch_views = int(input_ch_views)

    inputs = tf.keras.Input(shape=(input_ch_coord + input_ch_views + np.prod(input_ch_image),))
    inputs_pts, inputs_views, inputs_images = tf.split(inputs, [input_ch_coord, input_ch_views, np.prod(input_ch_image)], -1)
    inputs_pts.set_shape([None, input_ch_coord])
    inputs_views.set_shape([None, input_ch_views])
    print([None] + list(input_ch_image))
    print(inputs_images.shape)
    inputs_images = tf.reshape(inputs_images,[-1] + list(input_ch_image))
    inputs_images = tf.cast(inputs_images, tf.float32)

    print("The shapes are:")
    print(inputs.shape, inputs_pts.shape, inputs_views.shape, inputs_images)

    # feature_vector = inputs_images
    # feature_vector = conv2d(32,5,input_ch_image)(feature_vector)
    # feature_vector = maxpool((64,64))(feature_vector)

    # inception res v2
    feature_vector = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs_images)
    pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=input_ch_image)
    pretrained_model.trainable = False
    feature_vector = pretrained_model(feature_vector)
    feature_vector = tf.reshape(feature_vector,[-1,np.prod(feature_vector.shape[1:])])

    print("feature_vector shape is:")
    print(feature_vector.shape)

    # concate feature vector with input coordinates
    outputs = tf.concat([inputs_pts, feature_vector], -1)
    # outputs = inputs_pts

    print("outputs shape is:")
    print(outputs.shape)
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs_pts, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        # alpha is the density of target point
        bottleneck = dense(256, act=None)(outputs)
        inputs_viewdirs = tf.concat(
            [bottleneck, inputs_views], -1)  # concat viewdirs
        outputs = inputs_viewdirs
        for i in range(1):
            outputs = dense(W//2)(outputs)
        outputs = dense(3, act=None)(outputs)
        # this outputs is r,g,b of target point
        outputs = tf.concat([outputs, alpha_out], -1)
    else:
        outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def init_nerf_r_models(D=8, W=256, D_rotation=2, input_ch_image=(400, 400, 3), input_ch_pose=(3,4), input_ch_coord=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, feature_len=256, rot_mlp=False):
    # what is input ch views? -- input channel number for viewing direction
    # cos positional encoding is also put on viewing direction as well

    print("Initing nerf_r models")

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)
    def conv2d(filter, kernel_size, input_shape): return tf.keras.layers.Conv2D(filter, kernel_size, padding='valid', input_shape=input_shape)
    def maxpool(pool_size): return tf.keras.layers.MaxPool2D(pool_size)
    def avgpool(pool_size): return tf.keras.layers.AveragePooling2D(pool_size)

    input_ch_coord = int(input_ch_coord)
    input_ch_views = int(input_ch_views)

    # first model: inception + MLP as encoder
    inputs_encoder = tf.keras.Input(shape=(np.prod(input_ch_image) + np.prod(input_ch_pose),))
    inputs_images, input_poses = tf.split(inputs_encoder, [np.prod(input_ch_image), np.prod(input_ch_pose)], -1)
    inputs_images = tf.reshape(inputs_images,[-1] + list(input_ch_image))
    input_poses.set_shape([None, np.prod(input_ch_pose)])

    # res v2
    feature_vector = tf.keras.applications.resnet.preprocess_input(inputs_images)
    pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_ch_image, pooling='avg')
    pretrained_model.trainable = False
    feature_vector = pretrained_model(feature_vector)

    # apply MLP to do rotation
    if rot_mlp:
        feature_vector = tf.concat([feature_vector,input_poses], -1)
        for i in range(D_rotation):
            feature_vector = dense(W)(feature_vector)
    outputs_encoder = dense(feature_len, act=None)(feature_vector)

    model_encoder = tf.keras.Model(inputs=inputs_encoder, outputs=outputs_encoder)

    # -----------------------------------------------
    # Decoder MLP for predicting rbg and density
    inputs = tf.keras.Input(shape=(input_ch_coord + input_ch_views + feature_len,))
    inputs_pts, inputs_views, features = tf.split(inputs, [input_ch_coord, input_ch_views, feature_len], -1)
    inputs_pts.set_shape([None, input_ch_coord])
    inputs_views.set_shape([None, input_ch_views])
    features.set_shape([None, feature_len])

    # concate feature vector with input coordinates
    decoder_inputs = tf.concat([inputs_pts, features], -1)
    outputs = decoder_inputs

    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([decoder_inputs, outputs], -1)

    if use_viewdirs:
        alpha_out = dense(1, act=None)(outputs)
        # # alpha is the density of target point
        # bottleneck = dense(256, act=None)(outputs)
        # inputs_viewdirs = tf.concat(
        #     [bottleneck, inputs_views], -1)  # concat viewdirs
        # outputs = inputs_viewdirs
        # for i in range(1):
        #     outputs = dense(W//2)(outputs)
        # outputs = dense(3, act=None)(outputs)
        # # this outputs is r,g,b of target point
        outputs = alpha_out
    else:
        print("Error: must use viewdirs for nerf_r model!")
        return
    model_decoder = tf.keras.Model(inputs=inputs, outputs=outputs)

    return [model_encoder, model_decoder]

def init_pixel_nerf_encoder(input_ch_image=(200, 200, 3), feature_depth=256, add_global_feature=False, args=None):

    
    '''
    Init encoder from paper pixel nerf
    Uses a ResNet 50 pretrained auto-encoder, outputs feature from each layer concatenated

    Input:
        feature_depth: the depth of feature volume
        add_global_feature: whether to keep a global feature from last layer. This would have same size as feature_depth
    '''

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)


    # first model: resnet 50
    inputs_encoder = tf.keras.Input(shape=(np.prod(input_ch_image),))
    inputs_images = tf.reshape(inputs_encoder,[-1] + list(input_ch_image))

    inputs_images = tf.keras.applications.resnet.preprocess_input(inputs_images)
    pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_ch_image, pooling='avg')

    # ResNet18, preprocess_input = Classifiers.get('resnet18')
    # pretrained_model = ResNet18(include_top=False, input_shape=input_ch_image, pooling='avg', weights='imagenet')
    # inputs_images = preprocess_input(inputs_images)

    use_feature_volume = args.use_feature_volume if args is not None else True

    if use_feature_volume:
        # get features from middle layers
        mid_out_layers = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
        # mid_out_layers = ['pool1_pool','conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        # mid_out_layers = ['stage1_unit2_conv2', 'stage2_unit2_conv2', 'stage3_unit2_conv2', 'stage4_unit2_conv2']

        features = []    
        for layer in mid_out_layers:
            feature = pretrained_model.get_layer(layer).output
            features.append(feature)
        
        global_feature = pretrained_model.get_layer(mid_out_layers[-1]).output

        # rescale features to same size as input image
        HW = inputs_images.shape[1:3]
        for i, feature in enumerate(features):
            features[i] = tf.image.resize(feature, HW)
        features = tf.concat(features, axis=-1)
        pretrained_encoder = tf.keras.Model(inputs=pretrained_model.inputs, outputs=(features, global_feature), name='resnet50')

    else:
        pretrained_encoder = pretrained_model

    # trainable_encoder = not args.freeze_encoder if args is not None else True
    # pretrained_encoder.trainable = False
    # if trainable_encoder:
    #     # unfreeze top layers that are not batch norm
    #     unfreeze_from = args.unfreeze_from if args is not None else 0
    #     assert unfreeze_from <= len(pretrained_encoder.layers)
    #     for layer in pretrained_encoder.layers[unfreeze_from:]:
    #         if not layer.name.endswith('bn'):
    #             layer.trainable = True

    pretrained_encoder.trainable = not args.freeze_encoder if args is not None else True

    if pretrained_encoder.trainable:
        # freeze first few layers
        unfreeze_from = args.unfreeze_from if args is not None else 0
        for layer in pretrained_encoder.layers[:unfreeze_from]:
            layer.trainable = False

        # freeze batch normalization layers
        for layer in pretrained_encoder.layers:
            if layer.name.endswith('bn'):
                layer.trainable = False

    # run in inference mode to prevent updating the bactNorm layers
    # but this thing is causing weird behaviour!
    features, global_feature = pretrained_encoder(inputs_images, training=True)

    if not args.only_global_feature:
        # pass feature for each pixel to a MLP to shrink size
        features_original_shape = features.shape
        features = tf.reshape(features, [-1, features_original_shape[-1]])
        features = dense(feature_depth)(features)

        # reshape it to volume
        volume_shape = (features_original_shape[1:-1] + [feature_depth]).as_list()
        features = tf.reshape(features, [-1] + volume_shape)
    else:
        features = features[...,:1]

    # do the same thing for global feature
    if use_feature_volume and add_global_feature:
        global_feature = tf.keras.layers.AveragePooling2D(pool_size=global_feature.shape[1:3], strides=1)(global_feature)
        dim = np.prod(global_feature.shape[1:])    
        global_feature = tf.reshape(global_feature,[-1,dim])
        global_feature = dense(feature_depth)(global_feature)
        features = [features, global_feature]

    
    model_encoder = tf.keras.Model(inputs=inputs_encoder, outputs=features, name='encoder')

    return model_encoder

def init_pixel_nerf_decoder(D=8, W=256, input_ch_coord=3, input_ch_views=3, output_ch=4, skips=[2,4,6], feature_depth=256, mode='add'):
    '''
    Init decoder architecture from pixelNerf paper
    Uses skip connections for several times
    Take feature of shape (B, feature_depth) as input, feed that through a 128 MLP first
    '''

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)
    def conv2d(filter, kernel_size, input_shape): return tf.keras.layers.Conv2D(filter, kernel_size, padding='valid', input_shape=input_shape)
    def maxpool(pool_size): return tf.keras.layers.MaxPool2D(pool_size)
    def avgpool(pool_size): return tf.keras.layers.AveragePooling2D(pool_size)

    input_ch_coord = int(input_ch_coord)
    input_ch_views = int(input_ch_views)

    # -----------------------------------------------
    # Decoder MLP for predicting rbg and density
    inputs = tf.keras.Input(shape=(input_ch_coord + input_ch_views + feature_depth,))

    # note that this can be further simplified
    inputs_pts, inputs_views, features = tf.split(inputs, [input_ch_coord, input_ch_views, feature_depth], -1)
    inputs_pts.set_shape([None, input_ch_coord])
    inputs_views.set_shape([None, input_ch_views])
    features.set_shape([None, feature_depth])
    # concate feature vector with input coordinates
    decoder_inputs = tf.concat([inputs_pts,inputs_views, features], -1)

    if mode == 'add':
        # if using addition skip, need to ensure the shapes are the same
        decoder_inputs = dense(W)(decoder_inputs)
    elif mode == 'concat':
        decoder_inputs = dense(W//2)(decoder_inputs)
    else:
        print('skip type not known!')
        return

    decoder_inputs_len = decoder_inputs.shape[-1]
    outputs = decoder_inputs

    for i in range(D):
        if mode == 'add':
            outputs = dense(W)(outputs)
            if i in skips:
                outputs =tf.keras.layers.Add()([outputs, decoder_inputs]) #? divide by 2 actually works! -- might not be the case

        elif mode == 'concat':
            # use concate skip
            if i+1 in skips:
                # next layer has skip connection, need to shrink the size of this ouput
                outputs = dense(W-decoder_inputs_len)(outputs)
            else:
                outputs = dense(W)(outputs)
            
            if i in skips:
                outputs = tf.concat([outputs,decoder_inputs], -1)

    outputs = dense(output_ch, act=None)(outputs)

    model_decoder = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model_decoder

