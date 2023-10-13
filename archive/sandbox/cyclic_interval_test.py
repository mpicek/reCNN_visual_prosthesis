import torch
import numpy as np


rng = np.arange(-10, 10, 0.05)
for i in rng:
    print(i)
    print(torch.remainder((torch.Tensor([i]) + 1), 2) - 1)
