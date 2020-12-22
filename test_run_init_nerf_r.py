from run_nerf_helpers import *
import numpy as np


encoder = init_pixel_nerf_decoder()

encoder.summary()



# obj = np.load('data/ImageNet-ResNet18.npz', encoding='latin1')

# print(obj)

# print(model.trainable_variables)