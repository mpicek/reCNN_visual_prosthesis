from models import LOL
from ch_vs_rot import config
import torch
from Lurz_dataset import LurzDataModule
import numpy as np

data_dir = 'data/lurz2020/static20457-5-9-preproc0'
dataset_config = {"data_dir": data_dir, 
                      "batch_size": 10, 
                      "normalize": True, 
                      "exclude": "images"}
dm = LurzDataModule(**dataset_config)
dm.prepare_data()
dm.setup()

s = 20
k_rot = 2

config.update({"input_size_x": s, "input_size_y": s, "num_neurons": dm.get_output_shape()[0], "mean_activity": dm.get_mean()})
config.update({"input_channels": 1})
config.update({"core_layers": 2})
config.update({"core_hidden_channels": 3})
model = LOL(**config)

c_dim = config["core_hidden_channels"] * config["num_rotations"]

from neuralpredictors.layers.cores.conv2d import RotationEquivariant2dCore

core = RotationEquivariant2dCore(
    num_rotations=config["num_rotations"],
    stride=config["stride"],
    upsampling=config["upsampling"],
    rot_eq_batch_norm=config["rot_eq_batch_norm"],
    input_regularizer=config["input_regularizer"],
    input_channels=config["input_channels"],
    hidden_channels=config["core_hidden_channels"],
    input_kern=config["core_input_kern"],
    hidden_kern=config["core_hidden_kern"],
    layers=config["core_layers"],
    gamma_input=config["core_gamma_input"],
    gamma_hidden=config["core_gamma_hidden"],
    stack=config["stack"],
    depth_separable=config["depth_separable"],
    use_avg_reg=config["use_avg_reg"]
)

im = torch.rand((1, 1, s, s))
im = (im * 255).floor()
im_rot = torch.rot90(im[0, 0], k_rot, [0, 1])
im_rot = im_rot.unsqueeze(0).unsqueeze(0)


out = core(im)
out_rot = core(im_rot)

pairs = [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 0], [7, 1]]

def find(a, b, k=0, now=0):
    """
        Have features "a" and try to fit them to features "b",
        but previously rotate "a" by 90 * k counterclockwise.
    """
    a_rot = torch.rot90(a, k, [0, 1])
    # print(a_rot)
    for i in range(c_dim):
        print(torch.allclose(a_rot, b[0, i], rtol=1e-04, atol=1e-05)) #, rtol=1e-04, atol=1e-07
        # print(np.allclose(a_rot.detach().cpu().numpy(), b[0, i].detach().cpu().numpy(), rtol=1e-04, atol=1e-05))
        # print("----")
        for pair in pairs:
            if pair[0] == now and pair[1] == i:
        # if i == 5:
                pass
                # print(b[0,i])
                # print(a_rot - b[0,i])
                # print((a_rot - b[0,i]).max())

for i in range(c_dim):
    wanted = out[0,i]
    # if i == 3:
        # print(wanted)
    # print(torch.rot90(wanted, 1, [0, 1]))
    find(wanted, out_rot, k_rot, i)
    print("----------------------------------------------------------------------")
