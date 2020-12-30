# experimental run that takes two images + transformation matrix + 3D coordinate
# will only run on config_lego_rotation

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
from scipy import sparse
import imageio
import json
import random
import time

from utils.run_nerf_helpers import *
from utils.config_parser import config_parser
from utils.load_llff import load_llff_data
from utils.load_deepvoxels import load_dv_data
from utils.load_blender import load_blender_data
from utils.load_shapenet import load_shapenet_data

from model.embedder import *
from model.models import *

tf.compat.v1.enable_eager_execution()


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    # chunck = batch size
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def compute_features(input_image, input_pose, encoder):
    # flatten input image
    image_original_shape = list(input_image.shape)
    input_image = np.reshape(input_image,[1, -1])
    input_image = tf.cast(input_image, tf.float32)

    # flatten input pose and add it to inputs
    # for pixelNerf architecture, no input pose is used
    # input_pose = np.reshape(input_pose,[1, -1])
    # input_pose = tf.cast(input_pose, tf.float32)

    # inputs = tf.concat([input_image, input_pose], -1)
    feature = encoder(input_image)
    # feature = tf.reshape(feature, [1] + image_original_shape[:-1] + [feature.shape[-1]])
    return feature


def run_network(inputs, select_coords, input_image, input_pose, viewdirs, network_fn, embed_fn, embeddirs_fn, netchunk=1024*64, feature=None):
    """
    Prepares inputs and applies network 'fn'.

    Input:
        inputs: 3D coordinates in world space of sampled points, shape [B, S, 3] where B=#batch, S=#sampled points for a pixel
        select_coords: the 2D pixel coordinates of those points, shape [B, 2]

    
    """

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        # a different type of reshape
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # viewing direction is also embedded
        embedded = tf.concat([embedded, embedded_dirs], -1)

    if feature is None:
        feature = compute_features(input_image, input_pose, network_fn['encoder'])

    if len(feature.shape) > 2:
        # returned feature is a volume, take samples for each pixel
        # for each query point, sample corresponding feature
        select_coords = tf.expand_dims(select_coords,axis=1)
        select_coords = tf.tile(select_coords,[1, inputs.shape[1], 1])
        features_sampled = tf.gather_nd(feature[0,...],select_coords)
        features_sampled = tf.reshape(features_sampled,[-1, features_sampled.shape[-1]])
    else:
        # global feature vector
        features_sampled = tf.tile(feature, [embedded.shape[0],1])
    
    embedded = tf.concat([embedded, features_sampled], -1)

    # removed batchfy for pts here
    outputs_flat = network_fn['decoder'](embedded)
    outputs = tf.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    # last element in the output shape is the output channel
    outputs = tf.concat([outputs,outputs,outputs,outputs],axis=-1)
    return outputs, feature


def create_nerf(args, hwf):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 1
    skips = [2,4,6]
    # skip specifies the indices of layers that need skip connection

    encoder = init_pixel_nerf_encoder(input_ch_image=(hwf[0],hwf[1],3), feature_depth=args.feature_len, args=args)
    # feature_depth = encoder.outputs[0].shape[-1]

    decoder = init_pixel_nerf_decoder(D=args.netdepth, W=args.netwidth, 
                                      input_ch_coord=input_ch, 
                                      input_ch_views=input_ch_views, output_ch=output_ch, 
                                      skips=skips, feature_depth=args.feature_len, mode=args.skip_type)
    
    if args.fix_decoder:
        decoder.trainable = False
    models = {'encoder': encoder, 'decoder': decoder}

    encoder.summary()
    decoder.summary()

    # fine model: only fine decoder needed
    decoder_fine  = None
    if args.N_importance > 0 and args.separate_fine:
        decoder_fine = init_pixel_nerf_decoder(D=args.netdepth, W=args.netwidth, 
                                      input_ch_coord=input_ch, 
                                      input_ch_views=input_ch_views, output_ch=output_ch, 
                                      skips=skips, feature_depth=args.feature_len, mode=args.skip_type)
        if args.fix_decoder:
            decoder_fine.trainable = False
        models['decoder_fine'] = decoder_fine


    def network_query_fn(inputs, select_coords, input_image, input_pose, viewdirs, network_fn, feature=None): return run_network(
        inputs, select_coords, input_image, input_pose, viewdirs, network_fn, feature=feature,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': decoder_fine,
        'N_samples': args.N_samples,
        'network_fn': models,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    # start is batch counter
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('encoder' in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading encoder from', ft_weights)
        models['encoder'].set_weights(np.load(ft_weights, allow_pickle=True))
        ft_weights_decoder = ft_weights.replace('encoder', 'decoder')
        print('Reloading decoder from', ft_weights_decoder)
        models['decoder'].set_weights(np.load(ft_weights_decoder, allow_pickle=True))

        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        # load weights for fine decoder
        if decoder_fine is not None:            
            ft_weights_decoder_fine = ft_weights.replace('encoder', 'decoder_fine')
            print('Reloading decoder_fine from', ft_weights_decoder_fine)
            models['decoder_fine'].set_weights(np.load(ft_weights_decoder_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, models


def render_rays(ray_batch,
                select_coords,
                input_image,
                input_pose,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################

    input_pose = tf.convert_to_tensor(input_pose)
    input_image = tf.convert_to_tensor(input_image)
    
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # TODO: find out if this affects original nerf
    rays_d = ray_batch[:, 8:] # use normalized rays_d

    # Extract unit-normalized viewing direction.
    # note that viewdirs are same as rays_d, it's used here just for no_view_dir case
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1] 

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.

    # modified to exclude distance 0
    t_vals = tf.linspace(0., 1., N_samples+1)[1:]
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3

    def transform_coord_sys(vecs, input_pose, normalize=False):
        '''
        Transform 3D points or view dirs into input view's coordinate system

        Input:
            vecs: (B,S,3) vectors of 3D coordinates or view dirs. #S is the number of samples
            input_pose: input view's coordinate system of shape (3, 4)
            normalize: whether to normalize after transform. needed for transforming view dirs
        '''
        # convert those points from world coordinate to input view camera coordiate
        input_pose_full = tf.concat([input_pose,[[0,0,0,1]]], axis=0)

        vecs_input_view = tf.transpose(tf.reshape(vecs, (-1, 3)))
        vecs_input_view = tf.concat([vecs_input_view, tf.ones([1, vecs_input_view.shape[-1]])],axis=0)

        vecs_input_view = tf.linalg.inv(input_pose_full) @ vecs_input_view # TODO: make sure it works

        vecs_input_view = tf.transpose(vecs_input_view[:3,:])
        vecs_input_view = tf.reshape(vecs_input_view, vecs.shape)

        # # caculate correct view dirs in input view's coordinate
        # # seems to be incorrect
        # input_origin = vecs[:3, -1]
        # viewdirs = pts_input_view - input_origin[None, None, :]
        # viewdirs = viewdirs / tf.math.reduce_sum(viewdirs**2, axis=-1, keepdims=True)**0.5

        if normalize:
            vecs_input_view = vecs_input_view / tf.math.reduce_sum(vecs_input_view**2, axis=-1, keepdims=True)**0.5

        return vecs_input_view

    pts_input_view = transform_coord_sys(pts, input_pose)
    viewdirs_input_view = transform_coord_sys(viewdirs, input_pose, normalize=True)

    # Evaluate model at each point.
    raw, feature = network_query_fn(pts_input_view, select_coords, input_image, input_pose, viewdirs_input_view, network_fn)  # [N_rays, N_samples, 4]
    _, disp_map, acc_map, weights, rgb_map = raw2outputs(
        raw, z_vals, rays_d)

    rgb_map = tf.concat([rgb_map[:,None],rgb_map[:,None],rgb_map[:,None]],axis=-1)
    rgb_map = rgb_map/2.

    if N_importance > 0:
        # _0 are the coarse estimation if N_impoartance > 0
        rgb_map_0, disp_map_0, acc_map_0  = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]


        pts_input_view = transform_coord_sys(pts, input_pose)
        viewdirs_input_view = transform_coord_sys(viewdirs, input_pose, normalize=True)

        # Make predictions with network_fine if it exists
        run_fn = network_fn if network_fine is None else {'decoder': network_fine}
        # re-use the same feature
        raw, _ = network_query_fn(pts_input_view, select_coords, input_image, input_pose, viewdirs_input_view, run_fn, feature=feature)
        _, disp_map, acc_map, weights, rgb_map = raw2outputs(
            raw, z_vals, rays_d)
        
        rgb_map = tf.concat([rgb_map[:,None],rgb_map[:,None],rgb_map[:,None]],axis=-1)
        # normalize to fit the far/near bound
        rgb_map = rgb_map/2.

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'feature': feature}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def render(H, W, focal,
           chunk=1024*32, rays=None, select_coords=None, image=None, 
           pose=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      select_coords: pixel coordinates of selected pixels. Used to extract local feature.
        If None, assumes in order pixels covering the whole image

      pose: c2w tranformation matrix of input image

      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if select_coords is None:
        select_coords = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), -1)
        select_coords = tf.reshape(select_coords, [-1, 2])

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)


    def batchify_rays(rays_flat, select_coords, image, pose, chunk=1024*32, **kwargs):
        """Render rays in smaller minibatches to avoid OOM."""
        all_ret = {}
        for i in range(0, rays_flat.shape[0], chunk):
            ret = render_rays(rays_flat[i:i+chunk], select_coords[i:i+chunk], image, pose, **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
        return all_ret

    # Render and reshape
    all_ret = batchify_rays(rays, select_coords, image, pose, chunk, **kwargs)
    for k in all_ret:
        if 'feature' not in k:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'feature']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, chunk, render_kwargs, input_image=None, pose=None, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, feature, _ = render(
            H, W, focal, image=input_image, pose=pose, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def train():

    parser = config_parser()
    args = parser.parse_args()

    if not args.use_depth:
        print("ERROR! only depth is supported on this branch for now")
        return
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data: lego is blender type

    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip, args.quarter_res)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

    elif args.dataset_type == 'shapenet':
        sample_nums = (args.shapenet_train, args.shapenet_val, args.shapenet_test)
        images, poses, render_poses, hwf, i_split, obj_indices, obj_names, obj_split = load_shapenet_data(
                        args.datadir, resolution_scale=args.resolution_scale,
                        sample_nums=sample_nums, fix_objects=args.fix_objects, args=args)
        # images: (n.H,W,C) array containing all images
        # i_split: list of length 3, each element is a numpy array containing all the image ids for train/val/test
        # obj_indices: (n_obj,n_view) array, n_obj is the number of objects, n_view is number of view points/images for each object
        # obj_names: (n_obj) array containing the name of each object
        # obj_split: list of len 3, each element is a numpy array containing all the object ids for trainval/test
        
        print('Loaded shapenet', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # Matches with blender renderer depth clip
        near = 0.
        far = 3.

        if args.view_val:
            # drop all given vals, pick 0, 8, 16, 24 from each object in train
            picks = np.array([0,8,16,24],dtype=int)
            i_train = np.array([],dtype=int)
            i_val = np.array([],dtype=int)
            for obj_i in obj_split[0]:
                img_ids = np.array(obj_indices[obj_i])
                i_val = np.concatenate([i_val, img_ids[picks]])
                i_train = np.concatenate([i_train, np.delete(img_ids,picks)])
                obj_indices[obj_i] = np.delete(obj_indices[obj_i], picks)

        # log images
        dataimgdir = os.path.join(args.basedir, args.expname, 'data_preview')
        os.makedirs(dataimgdir, exist_ok=True)

        if sample_nums == (1,0,0):
            # single obj mode, log all images
            for i in i_train:
                imageio.imwrite(os.path.join(dataimgdir, '{}_{}_{}.png'.format('train', obj_names, i)), to8b(images[i]))
            for i in i_val:
                imageio.imwrite(os.path.join(dataimgdir, '{}_{}_{}.png'.format('val', obj_names, i)), to8b(images[i]))
        else:
            for i in range(len(obj_indices)):
                index = obj_indices[i][0]

                prefix = ''
                if i in obj_split[0]:
                    prefix = 'train'
                elif i in obj_split[1]:
                    prefix = 'val'
                elif i in obj_split[2]:
                    prefix = 'test'

                imageio.imwrite(os.path.join(dataimgdir, '{}_{}.png'.format(prefix, obj_names[i])), to8b(images[index]))

        

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # remove the alpha channel from image
    if args.white_bkgd and images[0].shape[-1] > 3:
        images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
    else:
        images = images[..., :3]

    if args.train_num is not None and args.train_num < len(i_train):
        i_train = i_train[:args.train_num]



    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, models = create_nerf(
        args, hwf)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('raw ray shape:')
        print(rays.shape)
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)

    N_iters = args.N_iters
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
        
    rays_rgb = np.reshape(rays_rgb,[-1, H, W, 3, 3])

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        if i >= args.i_train_encoder and models['encoder'].get_layer('resnet50').trainable == False:
            print('-----------------------------')
            print('Unfreezing pretrained encoder')
            print('-----------------------------')
            pretrained_model = models['encoder'].get_layer('resnet50')
            pretrained_model.trainable = True

            # freeze first few layers
            for layer in pretrained_model.layers[:args.unfreeze_from]:
                layer.trainable = False

            # freeze batch normalization layers
            for layer in pretrained_model.layers:
                if layer.name.endswith('bn'):
                    layer.trainable = False

            models['encoder'].summary()


        # Sample random ray batch
        if args.use_rotation:
            # in order to apply rotation equivariant, need to pick two images from train set
            # if training with multiple objects, this part of code needs to be further modifed

            if args.dataset_type == 'shapenet' and sample_nums != (1,0,0):
                # need to make sure two images are from same object
                # if using single object, same as lego data
                obj_i = np.random.choice(np.arange(0, args.shapenet_train), 1)[0]
                img_i, target_i = np.random.choice(obj_indices[obj_i], 2, replace=False)

            else:
                # single object mode
                img_i, target_i = np.random.choice(i_train, 2, replace=False)
                
            traget_rays_rgb_index = np.where(i_train==target_i)[0][0]
            # traget_rays_rgb_index is the index in rays_rgb
            # needed because rays_rgb only contain rays for train imgs
            input_img = images[img_i]
            target_img = images[target_i]
            input_pose = poses[img_i, :3, :4]
            target_pose = poses[target_i, :3, :4]

        else:
            # don't use rotation, but still input image + coordinates
            img_i = np.random.choice(i_train,1)[0]
            traget_rays_rgb_index = np.where(i_train==img_i)[0][0]
            input_img = images[img_i]
            target_img = images[img_i]
            input_pose = poses[img_i, :3, :4]
            target_pose = input_pose

        # select ray indices for training
        coords = tf.stack(tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'), -1)
        coords = tf.reshape(coords, [-1, 2])

        select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
        select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
        if N_rand == coords.shape[0]:
            # if use full image train, then don't shuffle
            select_inds = coords
        # input('Note that full image train without shuffling has been used!')

        # select rays using pose of target image
        batch = tf.gather_nd(rays_rgb[traget_rays_rgb_index], select_inds) # [N_rand,3,3]
        batch_rays, target_s = batch[:,:2,:], batch[:,2,:]
        batch_rays = tf.transpose(batch_rays,[1,0,2])

        #####  Core optimization loop  #####

        with tf.GradientTape(persistent=True) as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, feature, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays, select_coords=select_inds, image=input_img, pose=input_pose,
                verbose=i<10, retraw=True, **render_kwargs_train)



            # log rgb for debugging
            # testimgdir = os.path.join(basedir, expname, 'tboard_train_imgs')
            # if i==0:
            #     os.makedirs(testimgdir, exist_ok=True)
            # imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_train.png'.format(i)), to8b(tf.reshape(rgb,[60,60,3])))
            # imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_ground_truth_in_train.png'.format(i)), to8b(input_img))
            # imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_ground_truth_target_train.png'.format(i)), to8b(target_img))

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            # what is trans? -- [:,:,4] of the raw outputs from NN 
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                img_loss += img_loss0 # how does this work?
                psnr0 = mse2psnr(img_loss0)

            overall_loss = img_loss
            tape.watch(overall_loss)
            # Compute MSE for rotation
            rot_loss = tf.constant(0,dtype="float32")
            if args.use_rotation:
                feature_target = compute_features(target_img, target_pose, render_kwargs_train['network_fn']['encoder'])
                rot_loss = tf.keras.losses.MeanSquaredError()(feature_target, feature)

                # compute the sum of loss
                # overall_loss += rot_loss
                


        enc_vars = models['encoder'].trainable_variables
        gradients = tape.gradient(overall_loss, enc_vars)
        zips = list(zip(gradients, enc_vars))

        dec_vars = models['decoder'].trainable_variables
        if 'decoder_fine' in models:
            dec_vars += models['decoder_fine'].trainable_variables
        gradients = tape.gradient(overall_loss, dec_vars)
        zips += list(zip(gradients, dec_vars))
        optimizer.apply_gradients(zips)

        del tape # garbage collect the persistent tape

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0 and i > 0:
            for k in models:
                save_weights(models[k], k, i)


        if i % args.i_video == 0 and i > 0:
            viddir = os.path.join(basedir, expname, 'videos')
            os.makedirs(viddir, exist_ok=True)

            # generate video for test object
            # not needed to test time optimisation task
            if args.test_optimise_num <= 0:
                rgbs, disps = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test, input_image=target_img, pose=target_pose)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(
                    viddir, '{}_spiral_{:06d}_train_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4',
                                to8b(rgbs), fps=30, quality=8)
                # imageio.mimwrite(moviebase + 'disp.mp4',
                #                 to8b(disps / np.max(disps)), fps=30, quality=8)
                imageio.imwrite(os.path.join(viddir, '{:06d}_ground_truth_train.png'.format(i)), to8b(target_img))
            
            # generate video for val object
            img_i = np.random.choice(i_val)
            rgbs, disps = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test,input_image=images[img_i], pose=poses[img_i, :3, :4])
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                viddir, '{}_spiral_{:06d}_val_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4',
            #                  to8b(disps / np.max(disps)), fps=30, quality=8)
            imageio.imwrite(os.path.join(viddir, '{:06d}_ground_truth_val.png'.format(i)), to8b(images[img_i]))

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
            #     rgbs_still, _ = render_path(
            #         render_poses, hwf, args.chunk, render_kwargs_test,input_image=input_img)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4',
            #                      to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwf, args.chunk, render_kwargs_test, input_image=input_img,
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_log_target == 0:
            # Log a rendered train view to Tensorboard
            rgb, _, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=target_pose, image=input_img, pose=input_pose,
                                                    **render_kwargs_test)
            # Save out the validation image for Tensorboard-free monitoring
            testimgdir = os.path.join(basedir, expname, 'tboard_train_imgs')
            if i==0:
                os.makedirs(testimgdir, exist_ok=True)
            imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj.png'.format(i)), to8b(rgb))
            imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_ground_truth_in.png'.format(i)), to8b(input_img))
            imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_ground_truth_target.png'.format(i)), to8b(target_img))

            mse = img2mse(rgb, target_img)
            psnr = mse2psnr(mse)

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                tf.contrib.summary.image('rgb_train', to8b(rgb)[tf.newaxis])

                tf.contrib.summary.scalar('psnr_train_img', psnr)
                tf.contrib.summary.scalar('loss_train_img', mse)
                tf.contrib.summary.image('rgb_train_img', target_img[tf.newaxis])


        if i % args.i_print == 0 or i < 10 or i % args.i_img == 0:

            print(f'{expname}, iter {i}, psnr {psnr.numpy()}, img_loss {img_loss.numpy()}, rot_loss {rot_loss.numpy()}, global_step {global_step.numpy()}')
            print('iter time {:.05f}'.format(dt))

            print(f'feature average {np.average(np.abs(feature.numpy()))}')

            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', img_loss)
                tf.contrib.summary.scalar('rot_loss', rot_loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)

            if i % args.i_img == 0:
                # Log a rendered validation view to Tensorboard
                # if shapenet, val with all val objects
                img_is = []
                if args.dataset_type == 'shapenet' and len(obj_split[1]) > 0 and args.val_all:
                    for obj_i in obj_split[1]:
                        img_is.append((np.random.choice(obj_indices[obj_i]),obj_i))
                else:
                    img_is = [(np.random.choice(i_val), 0)]


                for img_i, obj_i in img_is:
                    target = images[img_i]
                    pose = poses[img_i, :3, :4]

                    rgb, disp, acc, feature, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, image=target, pose=pose,
                                                    **render_kwargs_test)

                    mse = img2mse(rgb, target)
                    psnr = mse2psnr(mse)
                    
                    # Save out the validation image for Tensorboard-free monitoring
                    testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                    if i==0:
                        os.makedirs(testimgdir, exist_ok=True)
                    imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_{}.png'.format(i, obj_i)), to8b(rgb))
                    imageio.imwrite(os.path.join(testimgdir, '{:06d}_obj_{}_ground_truth.png'.format(i, obj_i)), to8b(target))

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                        tf.contrib.summary.image('rgb_obj_{}'.format(obj_i), to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp_obj_{}'.format(obj_i), disp[tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'acc_obj_{}'.format(obj_i), acc[tf.newaxis, ..., tf.newaxis])

                        tf.contrib.summary.scalar('psnr_holdout_obj_{}'.format(obj_i), psnr)
                        tf.contrib.summary.scalar('loss_holdout_obj_{}'.format(obj_i), mse)
                        tf.contrib.summary.image('rgb_holdout_obj_{}'.format(obj_i), target[tf.newaxis])

                    if args.N_importance > 0:

                        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                            tf.contrib.summary.image(
                                'rgb0_obj_{}'.format(obj_i), to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image(
                                'disp0_obj_{}'.format(obj_i), extras['disp0'][tf.newaxis, ..., tf.newaxis])
                            tf.contrib.summary.image(
                                'z_std_obj_{}'.format(obj_i), extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)

if __name__ == '__main__':
    train()
