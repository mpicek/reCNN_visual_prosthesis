import torch
import torch.nn as nn
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve())) #.. this has to be the folder reCNN_visual_prosthesis
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def StandardizeClip(mean, std, pixel_min=None, pixel_max=None, dim=[1, 2, 3]):
    # module that uses Standardize and Clip, see below
    transf = [Normalize(mean=mean, std=std, dim=dim)]
    if pixel_min != None or pixel_max != None:
        transf.append(Clip(pixel_min=pixel_min, pixel_max=pixel_max))
    return nn.Sequential(*transf)

class Normalize(nn.Module):
    # module that normalizes the image to have mean and std
    def __init__(self, mean=None, std=None, dim=None, eps=1e-12):
        super().__init__()
        self.mean = mean
        self.std = std
        self.dim = dim
        self.eps = eps
    def forward(self, x, iteration=None):
        x_mean = x.mean(dim=self.dim, keepdims=True)
        target_mean = self.mean if self.mean is not None else x_mean
        x_std = x.std(dim=self.dim, keepdims=True)
        target_std = self.std if self.std is not None else x_std
        return target_std * (x - target_mean) / (x_std + self.eps) #+ target_mean

class Clip(nn.Module):
    # module that only clips values of the image
    def __init__(self, pixel_min, pixel_max):
        super().__init__()
        self.pixel_min = pixel_min
        self.pixel_max = pixel_max
    def forward(self, x):
        x = torch.clamp(x, min=self.pixel_min, max=self.pixel_max)
        return x

class TrainableImage(nn.Module):
    def __init__(self, img_size, n=1, c=1):
        super().__init__()
        self.img = nn.Parameter((torch.randn(n, c,*img_size)))
    def forward(self):
        return self.img

def plot_img(img, pixel_min, pixel_max):
    plt.figure(figsize = (8,8))
    if type(img) != np.ndarray:
        img = img.cpu().detach().squeeze().numpy()
    plt.imshow(img, cmap="gray", vmax=pixel_max, vmin=pixel_min)
    plt.colorbar()
    plt.show()

class GaborFilter(nn.Module):
    def __init__(self, min_pixel_intensity=-1.7, max_pixel_intensity=2, min_sigma=0.0000000001, 
                 init_pos_x=None, init_pos_y=None, init_ori=None, init_f=None, dataset_default_ori_shift=90):
        super().__init__()

        # initialize to large enough to cover the whole visual field, allowing the gradient
        # to flow from all positions, so that the model can decide on the best positions of neuron
        self.sigma_x = nn.Parameter(torch.ones(1) * 0.94) 
        self.sigma_y = nn.Parameter(torch.ones(1) * 0.94)

        if init_f is not None:
            self.f = nn.Parameter(torch.ones(1) * init_f)
        else:
            self.f = nn.Parameter(torch.rand(1))

        if init_ori is not None:
            # remainder because the dataset has 90 degrees default shift
            self.ori = nn.Parameter(torch.ones(1) * init_ori)
        else:
            self.ori = nn.Parameter(torch.rand(1))

        if init_pos_x is not None:
            self.pos_x = nn.Parameter(torch.ones(1) * init_pos_x)
        else:
            self.pos_x = nn.Parameter(torch.rand(1))

        if init_pos_y is not None:
            self.pos_y = nn.Parameter(torch.ones(1) * init_pos_y)
        else:
            self.pos_y = nn.Parameter(torch.rand(1))

        self.shift = nn.Parameter(torch.rand(1))
        self.scale = nn.Parameter(torch.ones(1))

        self.min_pixel_intensity = min_pixel_intensity
        self.max_pixel_intensity = max_pixel_intensity

        self.min_sigma = min_sigma

        x = np.linspace(-5.5, 5.5, 55)
        y = np.linspace(-5.5, 5.5, 55)
        self.meshgrid_x, self.meshgrid_y = np.meshgrid(x, y)
        self.meshgrid_x =  torch.from_numpy(self.meshgrid_x).float()
        self.meshgrid_y =  torch.from_numpy(self.meshgrid_y).float()

        self.meshgrid_x = self.meshgrid_x.to('cuda')
        self.meshgrid_y = self.meshgrid_y.to('cuda')
        self.dataset_default_ori_shift = dataset_default_ori_shift

    def forward(self, x):
        ori = torch.remainder(self.ori + ((self.dataset_default_ori_shift / 180) * np.pi), np.pi)
        meshgrid_x_rotated = self.meshgrid_x * torch.cos(ori) - self.meshgrid_y * torch.sin(ori)
        meshgrid_y_rotated = self.meshgrid_x * torch.sin(ori) + self.meshgrid_y * torch.cos(ori)

        rotated_pos_x = torch.cos(ori) * self.pos_x - torch.sin(ori) * self.pos_y
        rotated_pos_y = torch.sin(ori) * self.pos_x + torch.cos(ori) * self.pos_y

        gaussian_filter = torch.clip(self.scale, min=self.min_pixel_intensity, max=self.max_pixel_intensity) * torch.exp(
            -0.5
            * (
                (
                    torch.square(meshgrid_x_rotated - rotated_pos_x)
                    / (torch.square(self.sigma_x) + self.min_sigma) # + minimum_sigma as we don't want to divide by 0
                )
                + (
                    torch.square(meshgrid_y_rotated - rotated_pos_y)
                    / (torch.square(self.sigma_y) + self.min_sigma) # + minimum_sigma as we don't want to divide by 0
                )
            )
        )

        cosine_filter = torch.cos(
            2
            * np.pi
            * (meshgrid_x_rotated + rotated_pos_x)
            * self.f
            + torch.remainder(self.shift, 2*np.pi)
        )

        gabor_filter = gaussian_filter * cosine_filter
        out = torch.tensordot(x, gabor_filter)
        img_err = torch.square(x - gabor_filter)
        return out, gabor_filter, img_err

def train_mei(model, device, neuron, std=0.1, steps=500, lr=0.02):
    
    bar = tqdm(range(steps))

    min_pixel_value = -1.757
    max_pixel_value = 2.04
    my_clip = StandardizeClip((max_pixel_value + min_pixel_value) / 2, std, min_pixel_value, max_pixel_value)

    im = TrainableImage([55, 55])

    optim = torch.optim.Adam(im.parameters(), lr=lr)

    im.to(device=device)
    model.to(device=device)
    my_clip.to(device=device)
    im.train()
    model.eval()
    
    activation = None
    for step in bar:
        optim.zero_grad()
        activation = model(my_clip(im()))
        loss = -torch.mean(activation[:,neuron])
        loss.backward()
        optim.step()
        bar.set_description(f"activation = {-loss}")
    print("Last activation", torch.mean(activation[:,neuron]))

    # Testing the polarity (make an inverse MEI and see if the activation is still high)
    my_clip.eval()
    mean = my_clip(im()).mean()
    inversed_MEI = -(my_clip(im()) - mean) + mean
    inverse_activation = model(inversed_MEI)
    print("Inverse activation", torch.mean(inverse_activation[:,neuron]))
    return my_clip(im()).cpu().detach().squeeze().numpy()

def train_gabor(gabor, mei, device, steps=4000):
    gabor.to(device=device)

    optim = torch.optim.Adam(gabor.parameters(), lr=0.01)
    gabor.train()
    mei = torch.from_numpy(mei)
    mei = mei.to(device=device)

    gabor_bar = tqdm(range(steps))

    for step in gabor_bar:
        optim.zero_grad()
        err, gab, img_err = gabor(mei)
        loss = torch.mean(img_err)
        loss.backward()
        optim.step()
        gabor_bar.set_description(f"loss = {-loss}")

    _, gab, _ = gabor(mei)
    return gab.cpu().detach().squeeze().numpy()