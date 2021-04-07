# load images from cropped stanford car dataset
# the dataset only contains real life cars with no object/pose labels
# only for rendering
# the function pose_spherical is modified from original NeRF code

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
    # order of rotation is reversed here due to intristic/extrisict rotations
    return c2w


def fix_rotation(azimuth, elevation):
    return (90 + azimuth) * np.pi/180.0, (90 - elevation) * np.pi/180.0


def load_stanford_car(basedir='./data/stanford_car_cropped/', fix_H=64,fix_W=64, args=None):
    focal = 164  # look for blender default!

    imgs_dir = basedir
    renderings = [name for name in os.listdir(imgs_dir) if name.endswith('.jpg') or name.endswith('.png')]
    imgs = []


    for i, rendering_name in enumerate(renderings):
        img = imageio.imread(os.path.join(imgs_dir, rendering_name)).astype('float32') / 255
        
        # pad to square image
        H, W = img.shape[:2]
        img_size = max([H, W])

        square_img = np.ones((img_size, img_size, 3)).astype('float32')
        diff_H = (img_size - H) // 2
        diff_W = (img_size - W) // 2
        square_img[diff_H: H + diff_H,diff_W: W + diff_W, :] = img

        # resize
        square_img = tf.image.resize_area(square_img[None,...], [fix_H, fix_W]).numpy()
        # apply Gaussian blur
        # square_img = gaussian_filter(square_img[0,...], sigma=1)
        square_img = square_img[0,...]

        # reshape into square
        imgs.append(square_img)


    render_poses = tf.stack([pose_spherical(angle, 15, 1.0)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    # render poses for videos and render only experiments

    imgs = np.array(imgs)


    poses = np.array([pose_spherical(-180, args.render_elevation, 1.0)])
    poses = poses.astype(np.float32)

    return imgs, poses, render_poses, [fix_H, fix_W, focal]

