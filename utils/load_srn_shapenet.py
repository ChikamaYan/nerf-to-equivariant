import os
import numpy as np
import tensorflow as tf
import imageio
from scipy.ndimage import gaussian_filter
import glob
import tqdm

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

# what is the axis of rotation for those two?
# -- matrix form of the Euler rotations
# phi: rotation around object x axis (should be psi?)
rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

# theta: rotation around object y axis
rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


def load_srn_shapenet_data(basedir='../SRN_shapenet_car/', resolution_scale=0.5, sample_nums=(5, 0, 200), args=None):
    LARGE_LOAD = False

    if LARGE_LOAD:
        all_imgs = np.zeros([176704,64,64,3],dtype=np.float32) # hard coded!
    else:
        all_imgs = []
    all_poses = []

    train_obj_num, val_obj_num, test_obj_num = sample_nums

    if val_obj_num != 0:
        print('No validation is supported for SRN shapenet!')
        return

    train_dir = os.path.join(basedir,'cars_train')
    train_objs = sorted([obj_name for obj_name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, obj_name))])
    if train_obj_num == -1:
        train_obj_num = len(train_objs)
    train_objs = train_objs[:train_obj_num]

    test_dir = os.path.join(basedir,'cars_test')
    test_objs = sorted([obj_name for obj_name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, obj_name))])
    if test_obj_num == -1:
        test_obj_num = len(test_objs)
    test_objs = test_objs[:test_obj_num]

    focal = 131.25

    POSE_TRANS = np.diag([1,-1,-1,1])

    # generate indices for each object
    sample_counts = [0, train_obj_num, train_obj_num, train_obj_num + test_obj_num]
    obj_split = [np.arange(sample_counts[i], sample_counts[i+1])
                for i in range(3)] # tracks train/val/test object ids

    obj_indices = [] # tracks the image indices for each object
    next_img_i = 0 # track image indices

    for obj_names in [train_objs, test_objs]:
        if obj_names == train_objs:
            data_type_dir = train_dir
        else:
            data_type_dir = test_dir

        for obj in tqdm.tqdm(obj_names):
            rendering_path = os.path.join(data_type_dir, obj, 'rgb')
            renderings = [name for name in os.listdir(rendering_path)
                    if name.endswith('.png')]
            renderings.sort()

            pose_path = os.path.join(data_type_dir, obj, 'pose')
            poses = [name for name in os.listdir(pose_path)
                    if name.endswith('.txt')]
            poses.sort()

            imgs_indices = []

            for i, rendering_name in enumerate(renderings):
                imgs_indices.append(next_img_i)
                

                if LARGE_LOAD:
                    # resize early to fit in memory -- hard code mode
                    img = imageio.imread(os.path.join(rendering_path, rendering_name)).astype(np.float32)
                    H, W = img.shape[:2]
                    img = tf.image.resize_area(img[None,...], [int(H * resolution_scale), int(W * resolution_scale)]).numpy()[0,...]
                    
                    all_imgs[next_img_i,...] = (img/255.)[...,:3]
                else:
                    all_imgs.append(imageio.imread(os.path.join(rendering_path, rendering_name)))

                next_img_i += 1

                pose = np.loadtxt(os.path.join(pose_path, poses[i]))
                pose = np.reshape(pose, [4,4]) @ POSE_TRANS
                all_poses.append(pose)

            obj_indices.append(imgs_indices)
        

    obj_indices = np.array(obj_indices)
    i_split = [[],[],[]]


    # print(f'Objects for training are:{objs[obj_split[0]]}')
    # print(f'Objects for validation are:{objs[obj_split[1]]}')
    # print(f'Objects for testing are:{objs[obj_split[2]]}\n')

    # convert object indices in obj_split to i_split
    for i in range(len(obj_split)):
        i_split[i] = np.concatenate(obj_indices[obj_split[i]]) if len(obj_split[i]) > 0 else np.array([])

    # use same data for val and test
    i_split[1] = i_split[2]
    obj_split[1] = obj_split[2]

    # render poses for videos and render only experiments
    render_poses = tf.stack([pose_spherical(angle, -30.0, 1.3)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    # H, W = all_imgs[0].shape[:2]

    # all_imgs = np.concatenate(all_imgs)
    # all_imgs = tf.image.resize_area(all_imgs, [img_size, img_size]).numpy()

    if not LARGE_LOAD:
        H, W = all_imgs[0].shape[:2]

        H = int(H * resolution_scale)
        W = int(W * resolution_scale)
        all_imgs = np.array(all_imgs).astype(np.float32)
        all_imgs = tf.image.resize_area(all_imgs, [H, W]).numpy()
        all_imgs = all_imgs/255.

    focal = focal * resolution_scale

    
    all_poses = np.array(all_poses).astype(np.float32)

    all_objects = np.array(train_objs + test_objs)

    return all_imgs, all_poses, render_poses, [64, 64, focal], i_split, obj_indices, all_objects, obj_split

