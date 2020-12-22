import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json
# from classification_models.keras import Classifiers


# Misc utils

def img2mse(x, y): return tf.reduce_mean(tf.square(x - y))


def mse2psnr(x): return -10.*tf.math.log(x)/tf.math.log(10.)


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):

    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


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

def init_pixel_nerf_encoder(input_ch_image=(200, 200, 3)):

    
    '''
    Init encoder from paper pixel nerf
    Uses a ResNet 50 pretrained auto-encoder, outputs feature from each layer concatenated
    '''

    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu): return tf.keras.layers.Dense(W, activation=act)
    def conv2d(filter, kernel_size, input_shape): return tf.keras.layers.Conv2D(filter, kernel_size, padding='valid', input_shape=input_shape)
    def maxpool(pool_size): return tf.keras.layers.MaxPool2D(pool_size)
    def avgpool(pool_size): return tf.keras.layers.AveragePooling2D(pool_size)


    # first model: resnet 50
    inputs_encoder = tf.keras.Input(shape=(np.prod(input_ch_image),))
    inputs_images = tf.reshape(inputs_encoder,[-1] + list(input_ch_image))

    inputs_images = tf.keras.applications.resnet.preprocess_input(inputs_images)
    pretrained_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_ch_image, pooling='avg')

    # ResNet18, preprocess_input = Classifiers.get('resnet18')
    # pretrained_model = ResNet18(include_top=False, input_shape=input_ch_image, pooling='avg', weights='imagenet')
    # inputs_images = preprocess_input(inputs_images)

    pretrained_model.trainable = True

    # get features from middle layers
    mid_out_layers = ['pool1_pool','conv2_block3_out', 'conv3_block4_out']#, 'conv4_block6_out', 'conv5_block3_out']
    # mid_out_layers = ['stage1_unit2_conv2', 'stage2_unit2_conv2', 'stage3_unit2_conv2', 'stage4_unit2_conv2']

    features = []    
    for layer in mid_out_layers:
        feature = pretrained_model.get_layer(layer).output
        features.append(feature)

    # rescale features to same size as input image
    HW = inputs_images.shape[1:3]
    for i, feature in enumerate(features):
        features[i] = tf.image.resize(feature, HW)
    features = tf.concat(features, axis=-1)
    pretrained_part = tf.keras.Model(inputs=pretrained_model.input, outputs=features, name='pretrained_encoder')

    # run in inference mode to prevent updating the bactNorm layers
    features = pretrained_part(inputs_images, training=False)

    # pass feature for each pixel to a MLP to shrink size
    features_original_shape = features.shape
    features = tf.reshape(features, [-1, features_original_shape[-1]])
    features = dense(64)(features)
    
    model_encoder = tf.keras.Model(inputs=inputs_encoder, outputs=features, name='encoder')

    return model_encoder

def init_pixel_nerf_decoder(D=8, W=256, input_ch_pose=(3,4), input_ch_coord=3, input_ch_views=3, output_ch=4, skips=[2,4,6], feature_depth=128):
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
    inputs_pts, inputs_views, features = tf.split(inputs, [input_ch_coord, input_ch_views, feature_depth], -1)
    inputs_pts.set_shape([None, input_ch_coord])
    inputs_views.set_shape([None, input_ch_views])
    features.set_shape([None, feature_depth])

    # feed features to an MLP first
    # not needed -- done in encoder
    # features = dense(128)(features)

    # concate feature vector with input coordinates
    decoder_inputs = tf.concat([inputs_pts,inputs_views, features], -1)
    decoder_inputs_len = decoder_inputs.shape[-1]
    outputs = decoder_inputs

    for i in range(D):
        if i+1 in skips:
            # next layer has skip connection, need to shrink the size of this ouput
            outputs = dense(W-decoder_inputs_len)(outputs)
        else:
            outputs = dense(W)(outputs)
        
        if i in skips:
            outputs = tf.concat([outputs,decoder_inputs], -1)

    model_decoder = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model_decoder


# Ray helpers

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32),
                       tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = tf.stack([o0, o1, o2], -1)
    rays_d = tf.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling helper

def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights += 1e-5  # prevent nans
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = tf.linspace(0., 1., N_samples)
        u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [N_samples])
    else:
        u = tf.random.uniform(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = tf.searchsorted(cdf, u, side='right')
    below = tf.maximum(0, inds-1)
    above = tf.minimum(cdf.shape[-1]-1, inds)
    inds_g = tf.stack([below, above], -1)
    cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples
