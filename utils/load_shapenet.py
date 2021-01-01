import os
import numpy as np
import tensorflow as tf
import imageio
from scipy.ndimage import gaussian_filter
# from utils.load_blender import pose_spherical

# blender coord system
# z
# ^   ^ y
# |  /
# | /
# |/
# ---------> x

# OpenGL coord system
# y
# ^   ^ -z
# |  /
# | /
# |/
# ---------> x

# translation in z axis by t
trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_x = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_y = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,tf.sin(th),0],
    [0,1,0,0],
    [-tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_z = lambda th : tf.convert_to_tensor([
    [tf.cos(th),-tf.sin(th),0,0],
    [tf.sin(th),tf.cos(th),0,0],
    [0,0,1,0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(azimuth, elevation, radius):
    azimuth, elevation = fix_rotation(azimuth,elevation)
    # print(f'Rotations in xyz are: {elevation},{0},{azimuth}')
    c2w = trans_t(radius)
    c2w =  rot_z(azimuth) @ rot_y(0.0) @ rot_x(elevation) @ c2w
    # order of rotation is reversed here due to intristic/extrisict rotations -- not sure
    return c2w


def fix_rotation(azimuth, elevation):
    return (90 + azimuth) * np.pi/180.0, (90 - elevation) * np.pi/180.0


def load_shapenet_data(basedir='./data/shapenet/', resolution_scale=1., sample_nums=(5, 2, 1), fix_objects=None, args=None):
    SINGLE_OBJ = False

    all_imgs = []
    all_poses = []

    if args.use_depth:
        imgs_dir = os.path.join(basedir,'syn_depth')
    else:
        imgs_dir = os.path.join(basedir,'syn_rgb')


    if fix_objects is not None:
        print('Using specified objects')
        objs = np.array(fix_objects[0].split())
    else:
        objs = [obj_name for obj_name in os.listdir(imgs_dir)
                if os.path.isdir(os.path.join(imgs_dir, obj_name))]
        objs = np.random.choice(objs, np.sum(sample_nums), replace=False)

    focal = 210  # look for blender default!

    if sample_nums == (1, 0, 0):
        # signle object mode, doesn't allow i_test
        SINGLE_OBJ = True
        obj_split = [[], [], []]
        print('Using single object mode')
    else:
        sample_counts = [0, sample_nums[0], sample_nums[0] +
                         sample_nums[1], sum(sample_nums)]
        obj_split = [np.arange(sample_counts[i], sample_counts[i+1])
                   for i in range(3)]

    # tracks the indices for each object
    obj_indices = []

    for obj in objs:
        if args.use_depth:
            rendering_path = os.path.join(basedir,'syn_depth', obj)
        else:
            rendering_path = os.path.join(basedir,'syn_rgb', obj)
        renderings = [name for name in os.listdir(rendering_path)
                if name.endswith('.png')]
        renderings.sort()

        pose_path = os.path.join(basedir,'syn_pose', obj)
        poses = [name for name in os.listdir(pose_path)
                if name.endswith('.txt')]
        poses.sort()

        imgs_indices = []
        

        for i, rendering_name in enumerate(renderings):
            imgs_indices.append(len(all_imgs))
            all_imgs.append(imageio.imread(os.path.join(rendering_path, rendering_name)))
            pose = np.loadtxt(os.path.join(pose_path, poses[i]))
            all_poses.append(pose)
        obj_indices.append(imgs_indices)

    obj_indices = np.array(obj_indices)
    i_split = [[],[],[]]

    if SINGLE_OBJ:
        print(f'Object for training is:{objs}')

        if args.test_optimise_num > 0:
            print('TEST TIME OPTIMISATION EXPERIMENT')
            print(f'train = val = {args.test_optimise_num} images')
            input('Confirm?')

            i_split[1] = np.array(list(range(args.test_optimise_num)))
            i_split[0] = i_split[1].copy()
            i_split[2] = np.array([])
        else:
            i_split[1] = np.random.choice(list(range(len(all_imgs))),args.single_obj_val_num,replace=False)
            i_split[0] = np.array([i for i in range(len(all_imgs)) if i not in i_split[1]])
            i_split[2] = np.array([])


    else:
        print(f'Objects for training are:{objs[obj_split[0]]}')
        print(f'Objects for validation are:{objs[obj_split[1]]}')
        print(f'Objects for testing are:{objs[obj_split[2]]}\n')
        # convert object indices in obj_split to i_split

        for i in range(len(obj_split)):
            i_split[i] = np.concatenate(obj_indices[obj_split[i]]) if len(obj_split[i]) > 0 else np.array([])

    render_poses = tf.stack([pose_spherical(angle, 30.0, 1.0)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    # render poses for videos and render only experiments

    H, W = all_imgs[0].shape[:2]

    H = int(H * resolution_scale)
    W = int(W * resolution_scale)
    focal = focal * resolution_scale
    all_imgs = np.array(all_imgs).astype(np.float32)
    if args.use_depth:
        # expand 1 channel depth value to 3 channel rgb values
        all_imgs = np.stack([all_imgs,all_imgs,all_imgs],axis=-1) 
    all_imgs = tf.image.resize_area(all_imgs, [H, W]).numpy()

    if resolution_scale < 0.4:
        # apply Gaussian blur
        for i in range(all_imgs.shape[0]):
            all_imgs[i,...] = gaussian_filter(all_imgs[i,...], sigma=1)

    
    all_imgs = all_imgs/255.
    all_poses = np.array(all_poses)
    all_poses = all_poses.astype(np.float32)

    return all_imgs, all_poses, render_poses, [H, W, focal], i_split, obj_indices, objs, obj_split

