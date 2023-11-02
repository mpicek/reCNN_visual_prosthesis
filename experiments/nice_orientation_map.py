import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

import wandb
import glob
import pytorch_lightning as pl
from src.models import reCNN_bottleneck_CyclicGauss3d_no_scaling
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns # pip install seaborn

plt.rcParams.update({'font.size': 12})

def pickle_read(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x

img_path = "img/"

factor = 5.5

def circular_distance(a, b, period=np.pi):
    """
    Computes circular distance between two angles a and b.
    Changing period it's possible to set the scale (radiants or degrees)
    """
    return np.array(
        [min((aa - bb) % period, (bb - aa) % period) for aa, bb in zip(a, b)]
    )


def rotate_and_cumpute_distances(a, b, period, n_rot=360):
    """
    Rotates array of angles a for n_rot possible angles and compute average
    pairwise circular distance with respect to array of angles b.
    Changing period it's possible to set the scale (radiants or degrees)
    Return array containing average distances for all possible rotations and array of rotations
    """
    avg_dist = [
        circular_distance(a + r, b, period).mean()
        for r in np.linspace(0, period, n_rot)[:-1]
    ]
    return [np.array(avg_dist), np.linspace(0, period, n_rot)[:-1]]

def visualize(figsize=10, xlim=None, ylim=None, suffix=""):

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gcf().set_size_inches(7.5, 6.5)
    my_cmap = sns.color_palette("hls", as_cmap=True)
    # my_cmap = sns.palplot(sns.hls_palette(as_cmap=True))

    # my_cmap = sns.light_palette("Navy", as_cmap=True)
    points = ax.scatter(x, y, c=z, s=20, cmap='hsv') # plasma before, then twilight
    # points = ax.scatter(x, y, c=z, s=14, cmap="hsv",edgecolors='black') # plasma before, then twilight
    

    if suffix == "_antolik" or suffix == "_truth":
        ax.set_xlabel("Spatial dimension x [degrees of visual angle]")
        ax.set_ylabel("Spatial dimension y [degrees of visual angle]")

    else:
        plt.title("A location and preferred orientation estimates of a mouse\nrecorded in the Lurz et al. dataset\n")
        ax.set_xlabel("Normalized spatial dimension x")
        ax.set_ylabel("Normalized spatial dimension y")


    file_name = "nice_orientation_map" + suffix
    if xlim:
        plt.xlim(-xlim, xlim)
    if ylim:
        plt.ylim(-ylim, ylim)
    
    plt.axis('off')
    plt.savefig(img_path + file_name, dpi = 300, transparent=True, bbox_inches='tight', pad_inches=0)
    print("Reconstructed orientation maps are saved in " + file_name + ".fig")

    plt.clf()


run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")
pl.seed_everything(42)

artifact = run.use_artifact(
    "csng-cuni/reCNN_visual_prosthesis/model-hl2gbqtm:v0", type="model"
)

artifact_dir = artifact.download()
models_paths_list = glob.glob(artifact_dir + "/*.ckpt")
m = None
for path in models_paths_list:

    m = reCNN_bottleneck_CyclicGauss3d_no_scaling.load_from_checkpoint(path)
    m.freeze()

    print(f"Model from {path} loaded!")

squeezed = m.readout.mu.squeeze(0).squeeze(0).squeeze(1)

data_to_be_plotted = squeezed.cpu().detach().numpy()

x, y, z = data_to_be_plotted[:, 0], data_to_be_plotted[:, 1], data_to_be_plotted[:, 2]
z = (np.mod((z + 1), 2) ) -1

import random
indices = range(30000)
sample = random.sample(indices, 30000)


def pickle_save(path, x):
    with open(path, "wb") as f:
        pickle.dump(x, f)



artifact = run.use_artifact(
    "csng-cuni/reCNN_visual_prosthesis/model-abwwf3vs:v0", type="model"
)
artifact_dir = artifact.download()
print(artifact_dir)
models_paths_list = glob.glob(artifact_dir + "/*.ckpt")
m = None
for path in models_paths_list:
    m = reCNN_bottleneck_CyclicGauss3d_no_scaling.load_from_checkpoint(path)
    m.freeze()
    print(f"Model from {path} loaded!")

squeezed = m.readout.mu.squeeze(0).squeeze(0).squeeze(1)
data_to_be_plotted = squeezed.cpu().detach().numpy()
x, y, z = data_to_be_plotted[:, 0], data_to_be_plotted[:, 1], data_to_be_plotted[:, 2]
z = (np.mod((z + 1), 2) ) -1

x *= factor
y *= factor
new_z = [e if e >= 0 else e+1 for e in z]
z = new_z




pos_dict = pickle_read("data/antolik/position_dictionary.pickle")
positions = np.concatenate([pos_dict['V1_Exc_L2/3'].T, pos_dict['V1_Inh_L2/3'].T])
o_dict = pickle_read("data/antolik/oris.pickle")
ori = np.concatenate([np.array(o_dict['V1_Exc_L2/3']), np.array(o_dict['V1_Inh_L2/3'])])


pred = np.array(z)
pred = (2 * np.pi * (pred + 1) / 2) % np.pi
target = ori
result1 = rotate_and_cumpute_distances(pred, target, period=np.pi, n_rot=36)

pred = 180 * pred / np.pi
target = 180 * target / np.pi
result2 = rotate_and_cumpute_distances(pred, target, period=180, n_rot=36)


f, ax = plt.subplots()

print("Average error:")
print(result2[0][np.argmin(result2[0])])

print("Shift:")
print(result2[1][np.argmin(result2[0])])
shift = result2[1][np.argmin(result2[0])]

distances = circular_distance(((pred + shift) % 180), target, 180) 

z = (pred + shift) % 180

print(x.shape)
print(y.shape)
print(z.shape)
x = x[sample]
y = y[sample]
z = z[sample]
visualize(12, 2.55, 2.55, "_antolik")



z = target

positions = positions
x = positions[:, 0]
y = positions[:, 1]
y = -y

f, ax = plt.subplots()

print(x.shape)
print(y.shape)
print(z.shape)
x = x[sample]
y = y[sample]
z = z[sample]
visualize(12, 2.5, 2.5, "_truth")
