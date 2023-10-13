import torch
from torch.nn import functional as F
import numpy as np

N = 1
rotations = 4
h = 3
w = 5
neurons = 1

x = torch.Tensor(np.ones([1, rotations, h, w]))
x[:,0,:,:] = x[:,0,:,:] * 0
x[:,1,:,:] = x[:,1,:,:] * 90
x[:,2,:,:] = x[:,2,:,:] * 180
x[:,3,:,:] = x[:,3,:,:] * 270

y = torch.cat([x, torch.ones([1, 1, h, w])*360], dim=1)

# x = x.view(N, 1, rotations, h, w)
y = y.view(N, 1, rotations+1, h, w)
grid = torch.Tensor(np.ones([1, 1, neurons, 1, 3]))

rng = np.linspace(-1.2, 1.2, 100)
for i in rng:
    print(f"Value: {i}, rotation: ", end="")
    # kdyz zmenim zeros na border, tak to bude brat furt tu koncovou hodnotu

    print(F.grid_sample(y, grid * i, align_corners=True, mode="bilinear", padding_mode="border")[0,0,0,0,0])
    # print(F.grid_sample(x, grid * i, align_corners=False, mode="nearest", padding_mode="zeros")[0,0,0,0,0])

# print(F.grid_sample(x, grid , align_corners=False, mode="nearest", padding_mode="zeros")[0,0,0,0,0])