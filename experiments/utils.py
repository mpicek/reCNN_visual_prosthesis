import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve())) #.. this has to be the folder reCNN_visual_prosthesis
import wandb
import glob
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d
import matplotlib.pyplot as plt
import numpy as np
import pickle

def pickle_read(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x

def pickle_save(path, x):
    with open(path, "wb") as f:
        pickle.dump(x, f)

def circular_distance(a, b, period=np.pi):
    """
    Computes circular distance between two angles a and b.
    Changing period it's possible to set the scale (radiants or degrees)
    Written by Luca Baroni
    """
    return np.array(
        [min((aa - bb) % period, (bb - aa) % period) for aa, bb in zip(a, b)]
    )

def rotate_and_compute_distances(a, b, period, n_rot=360):
    """
    Rotates array of angles a for n_rot possible angles and computes average
    pairwise circular distance with respect to array of angles b.
    Changing period it's possible to set the scale (radiants or degrees)
    Return array containing average distances for all possible rotations and array of rotations
    Written by Luca Baroni
    """
    avg_dist = [
        circular_distance(a + r, b, period).mean()
        for r in np.linspace(0, period, n_rot)[:-1]
    ]
    return [np.array(avg_dist), np.linspace(0, period, n_rot)[:-1]]

def reconstruct_orientation_maps(x, y, ori, f, ax, save, figsize=10, xlim=None, ylim=None, img_path="img/", suffix="", neuron_dot_size=1):
    """
        Reconstructs the orientation map, visualizes it and prints the distribution of 
        orientation preferences.
    """

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gcf().set_size_inches(7.5, 6.5)
    points = ax.scatter(x, y, c=ori, s=neuron_dot_size, cmap="hsv")
    
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

    file_name = ""
    if suffix == "_truth":
        file_name = img_path + "true_orientation_map_of_antolik_model"
    else:
        file_name = img_path + "reconstructed_orientation_map" + suffix
    if xlim:
        plt.xlim(-xlim, xlim)
    if ylim:
        plt.ylim(-ylim, ylim)
    
    if save:
        plt.savefig(file_name, dpi = 300)
        print("Reconstructed orientation maps are saved in " + file_name + ".fig")

    plt.show()


def visualize_preferred_orientations(x, y, ori, f, ax, save, figsize=10, xlim=None, ylim=None, img_path="img/", suffix=""):

    plt.clf()
    plt.gcf().set_size_inches(6.5, 6.5)

    n, bins, patches = plt.hist(ori, bins=20)

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
    if save:
        plt.savefig(img_path + "distribution_of_prefered_orientations" + suffix, dpi = 300)
        print("saved")
    plt.show()
    plt.clf()

def download_model(artifact_name):
    run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")
    pl.seed_everything(42)

    artifact = run.use_artifact(
        artifact_name, type="model"
    )

    artifact_dir = artifact.download()
    models_paths_list = glob.glob(artifact_dir + "/*.ckpt")

    m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(models_paths_list[0])
    m.freeze()
    print(f"Model from {models_paths_list[0]} loaded!")

    return m

def get_neuron_estimates(model, scale_factor=1):
    squeezed = model.readout.mu.squeeze(0).squeeze(0).squeeze(1)

    data_to_be_plotted = squeezed.cpu().detach().numpy()

    x, y, ori = data_to_be_plotted[:, 0], data_to_be_plotted[:, 1], data_to_be_plotted[:, 2]

    # our model represents locations on positions from -1 and 1 (in both x and y 
    # directions), so we have to scale it to the real values
    #   - these values are given by the Antolik's model's characteristics
    x *= scale_factor
    y *= scale_factor

    # orientations are between -1 and 1 but there can be outliers that (because of
    # periodicity) have to be shifted to the right location
    # Example: 1.2 goes to -0.8, -1.7 goes to 0.3
    ori = (np.mod((ori + 1), 2) ) - 1 

    # we have orientations in all 360 degrees, but in this evaluation it makes sense
    # to treat bars of ori and ori+180 as bars of the same orientation
    # therefore, now we have orientations in a range [0, 1]
    ori = [e if e >= 0 else e+1 for e in ori]

    return (x, y, ori)

