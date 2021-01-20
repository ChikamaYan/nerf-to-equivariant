

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--train_num", type=int, default=None,
                        help='number of views used for training')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--skip_type", type=str, default='concat',
                        help='type of skip connection in decoder. Can be either add or concat')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_object", type=int, default=1,
                        help='batch size (number of objects per gradient step) note that for each object #N_rand rays will be drawn')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--freeze_encoder", action='store_true',
                        help='freeze the weights in the pretrained encoder')
    parser.add_argument("--unfreeze_from", type=int, default=150,
                        help='number of bottom layers kept untrainable')
    parser.add_argument("--query_z_only", action='store_true',
                        help='only use z coordinate (depth) to query local feature')
    
    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--separate_fine",  action='store_true',
                        help='whether to use separate models for fine sampling')                  
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)') # should be 3D
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--quarter_res", action='store_true',
                        help='load blender synthetic data at 200x200 instead of 800x800')
    parser.add_argument("--resolution_scale",  type=float, default='1.0',
                        help='apply resolution scale to loaded images')
                        

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_log_target",   type=int, default=99999999999,
                        help='frequency of generating a image for a target image from training dataset')
    parser.add_argument("--N_iters", type=int, default=200000,
                        help='number of iterations for training')
    parser.add_argument("--i_train_encoder", type=int, default=50000,
                        help='number of iterations for starting training the encoder')

    # rotation equivariant option
    parser.add_argument("--use_rotation", action='store_true',
                        help='use target+source equivariant for training')
    parser.add_argument("--use_rot_mlp", action='store_true',
                        help='use rotational MLP to rotate feature')
    parser.add_argument("--rot_mlp_depth", type=int, default=2,
                        help='depth of rotational MLP to rotate feature')
    parser.add_argument("--feature_len", type=int, default=256,
                        help='length of feature vector extracted from image')
    parser.add_argument("--use_feature_volume", action='store_true',
                    help='use feature volume + local feature for decoding')


    # shapenet options
    parser.add_argument("--shapenet_train", type=int, default=5,
                        help='number of shapenet objects used to train')
    parser.add_argument("--shapenet_val", type=int, default=2,
                        help='number of shapenet objects used to validate')
    parser.add_argument("--shapenet_test", type=int, default=1,
                        help='number of shapenet objects used to test')
    parser.add_argument("--fix_objects", type=str, nargs='+', default=None,
                        help='use specified objects')
    parser.add_argument("--view_val", action='store_true',
                        help='train with all train_objs, but val using different views of the same objs')
    parser.add_argument("--val_all", action='store_true',
                        help='during validation, use all val objects')
    parser.add_argument("--single_obj_val_num", type=int, default=5,
                        help='for single object mode, controls number of images reserved for validation')

    parser.add_argument("--use_depth", action='store_true',
                        help='use depth map as supervision')
    parser.add_argument("--fix_decoder", action='store_true',
                        help='fix the weights of decoder')
    parser.add_argument("--test_optimise_num", type=int, default=0,
                    help='number of images for test time optimisation. 0 means no test opt')
    parser.add_argument("--description", type=str, default='',
                    help='a description of the experiment, has no effect on algorithm')

    return parser

