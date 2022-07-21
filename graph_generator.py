import wandb
import glob
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 12})

def pickle_read(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x

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
    points = ax.scatter(x, y, c=z, s=1, cmap="hsv") # plasma before, then twilight
    
    if suffix == "_truth":
        plt.title("An orientation map of the Antolík et al. model\n")
    
    if suffix == "_antolik":
        plt.title("An orientation map of the Antolík et al. model\nreconstructed by our DNN\n")

    if suffix == "_antolik" or suffix == "_truth":
        ax.set_xlabel("Spatial dimension x [degrees of visual angle]")
        ax.set_ylabel("Spatial dimension y [degrees of visual angle]")
        f.colorbar(points).set_label("Preferred orientation [degrees]")
    else:
        plt.title("A location and preferred orientation estimates of a mouse\nrecorded in the Lurz et al. dataset\n")
        ax.set_xlabel("Normalized spatial dimension x")
        ax.set_ylabel("Normalized spatial dimension y")
        f.colorbar(points).set_label("Preferred orientation [degrees]")

    file_name = "reconstructed_orientation_maps_best" + suffix
    if xlim:
        plt.xlim(-xlim, xlim)
    if ylim:
        plt.ylim(-ylim, ylim)
    plt.savefig(file_name, dpi = 300)
    print("Reconstructed orientation maps are saved in " + file_name + ".fig")

    plt.clf()
    plt.gcf().set_size_inches(6.5, 6.5)

    n, bins, patches = plt.hist(z, bins=20)

    cm = plt.cm.get_cmap('hsv')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]
    col = bin_centers - min(bin_centers)
    col /= max(col)

    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    if suffix == "_truth":
        plt.title("A distribution of preferred orientations in Antolík et al. model\n")
    
    elif suffix == "_antolik":
        plt.title("A distribution of estimated preferred orientations\nin Antolík et al. model\n")

    else:
        plt.title("A distribution of estimated preferred orientations\nin the Lurz et al. dataset\n")

    plt.xlabel("Preferred orientation [degrees]")
    plt.ylabel("Number of neurons")
    plt.savefig("distribution_of_prefered_orientations_best" + suffix, dpi = 300)
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

    m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(path)
    m.freeze()

    print(f"Model from {path} loaded!")

squeezed = m.readout.mu.squeeze(0).squeeze(0).squeeze(1)

data_to_be_plotted = squeezed.cpu().detach().numpy()

x, y, z = data_to_be_plotted[:, 0], data_to_be_plotted[:, 1], data_to_be_plotted[:, 2]
z = (np.mod((z + 1), 2) ) -1

def pickle_save(path, x):
    with open(path, "wb") as f:
        pickle.dump(x, f)

new_z = [e if e >= 0 else e+1 for e in z]
z = new_z
z = [180*e for e in z]
f, ax = plt.subplots()
visualize(12, 1, 1, "_lurz")



artifact = run.use_artifact(
    "csng-cuni/reCNN_visual_prosthesis/model-abwwf3vs:v0", type="model"
)
artifact_dir = artifact.download()
print(artifact_dir)
models_paths_list = glob.glob(artifact_dir + "/*.ckpt")
m = None
for path in models_paths_list:
    m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(path)
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
visualize(12, 2.4, 2.4, "_antolik")



z = target

positions = positions
x = positions[:, 0]
y = positions[:, 1]
y = -y

f, ax = plt.subplots()
visualize(12, 2.4, 2.4, "_truth")

avg_error = np.mean(distances)
plt.gcf().set_size_inches(7.5, 6.5)
plt.hist(distances, bins=100, color="crimson")
plt.axvline(avg_error, color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(avg_error*1.1, max_ylim*0.9, 'Average error: {:.2f} degrees'.format(avg_error))
plt.xlim(0, 90)
plt.title("A distribution of errors in predicted orientation preference\n")
plt.xlabel("Error [degrees]")
plt.ylabel("Number of neurons")
plt.savefig("distribution_of_orientation_errors", dpi = 300)
plt.clf()