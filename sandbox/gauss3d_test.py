from models import Gaussian3dCyclic
from simple_train_picek import config
import torch
# from Lurz_dataset import LurzDataModule
import numpy as np
# from model_trainer import Lurz_dataset_preparation_function

# self,
#         in_shape,
#         outdims,
#         bias,
#         init_mu_range=0.5,
#         init_sigma_range=0.5,
#         batch_sample=True,
#         align_corners=True,
#         fixed_sigma=False,
#         mean_activity=None,
#         feature_reg_weight=1.0,
#         gamma_readout=None,  # depricated, use feature_reg_weight instead
#         **kw

s = 32 # size of images ... (s, s)
h = 6
w = 8
rotations = 4

x = torch.Tensor(np.ones([2, rotations, h, w]))
for i in range(rotations):
    x[:,i,:,:] = x[:,i,:,:] * (i+1) # channels (=rotations) have values 1, 2, .. rotations

im = x.floor()
# im = torch.rand((1, rotations, s, s)) #batchsize, rotations (not channels), w, h
# im = (im * 255).floor()
print("Image shape:", end=" ")
print(im.shape)
g = Gaussian3dCyclic((rotations, h, w), 15, False)
# g = Gaussian2d((rotations, h, w), 15, False)



a = g(im)
print(a)










# dm = Lurz_dataset_preparation_function(config)

# ############ WARNING ############
# # THE RotEqBottleneckGauss NEEDS TO HAVE THIS FORWARD FUNCTION:
#     # def forward(self, x):
#     #     x = self.core(x)
#     #     return x
# # IT IS BECAUSE WE DON'T WANT TO PASS IT THROUGH GAUSS TO SEE THE EQUIVARIANCE


# s = 4 # size of images ... (s, s)
# k_rot = 1 # how many times should I rotate the image by 90 deg (0, 1, 2, 3)

# config.update({"bottleneck_kernel": 3})
# config.update({"input_size_x": s, "input_size_y": s, "num_neurons": dm.get_output_shape()[0], "mean_activity": dm.get_mean()})
# config.update({"input_channels": 1})
# config.update({"core_layers": 3})
# config.update({"num_rotations": 8})
# config.update({"core_hidden_channels": 3})
# config.update({"batch_size": 1})

# model = RotEqBottleneckTest_NoGauss(**config)

# print("===========================================================")
# print(model.core.outchannels)

# im = torch.rand((1, 1, s, s))
# im = (im * 255).floor()
# im_rot = torch.rot90(im[0, 0], k_rot, [0, 1])
# im_rot = im_rot.unsqueeze(0).unsqueeze(0)


# out = model(im)
# out_rot = model(im_rot)

# c_dim = config["num_rotations"]

# def find(a, b, k=0, now=0):
#     """
#         Have features "a" and try to fit them to features "b",
#         but previously rotate "a" by 90 * k counterclockwise.
#     """
#     a_rot = torch.rot90(a, k, [0, 1])
#     # print(a_rot)
#     for i in range(c_dim):
#         print(torch.allclose(a_rot, b[0, i], rtol=1e-03, atol=1e-04)) #, rtol=1e-04, atol=1e-07


# # try all feature maps of out tensor and for each, try to find it in 
# # rotated image and passed through network (out_rot) and print on which position
# # we found it (with Trues and Falses)
# for i in range(c_dim):
#     wanted = out[0,i]
#     find(wanted, out_rot, k_rot, i)
#     print("----------------------------------------------------------------------")
