import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))

import wandb
import glob
import pytorch_lightning as pl
from src.models import reCNN_bottleneck_CyclicGauss3d
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({'font.size': 12})
factor = 5.5
img_path = "img/"

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
    Rotates array of angles a for n_rot possible angles and compute average
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

def visualize(figsize=10, xlim=None, ylim=None, suffix=""):
    """
        Reconstructs the orientation map, visualizes it and prints the distribution of 
        orientation preferences.
    """

    plt.gca().set_aspect("equal", adjustable="box")
    plt.gcf().set_size_inches(7.5, 6.5)
    points = ax.scatter(x, y, c=ori, s=1, cmap="hsv") # plasma before, then twilight
    
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
    plt.savefig(file_name, dpi = 300)
    print("Reconstructed orientation maps are saved in " + file_name + ".fig")

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
    plt.savefig(img_path + "distribution_of_prefered_orientations" + suffix, dpi = 300)
    plt.clf()


############################## ON LURZ'S DATASET ###############################

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

x, y, ori = data_to_be_plotted[:, 0], data_to_be_plotted[:, 1], data_to_be_plotted[:, 2]
ori = (np.mod((ori + 1), 2) ) -1

ori = [e if e >= 0 else e+1 for e in ori]
ori = [180*e for e in ori]
f, ax = plt.subplots()
visualize(12, 1, 1, "_lurz")


############################ ON ANTOLIK'S DATASET ##############################

# run the network
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
x, y, ori = data_to_be_plotted[:, 0], data_to_be_plotted[:, 1], data_to_be_plotted[:, 2]

# orientations are between -1 and 1 but there can be outliers that (because of
# periodicity) have to be shifted to the right location
ori = (np.mod((ori + 1), 2) ) -1 

# our model represents locations on positions from -1 and 1 (in both x and y 
# directions), so we have to scale it to the real values
x *= factor
y *= factor

# we have orientations in all 360 degrees, but in this evaluation it makes sense
# to treat bars of ori and ori+180 as bars of the same orientation
# therefore, now we have orientations in a range [0, 1]
ori = [e if e >= 0 else e+1 for e in ori]


#####################################################################################
################################## Position errors ##################################
#####################################################################################
# get the target values (values from the Antolik's computational model)
pos_dict = pickle_read("data/antolik/position_dictionary.pickle")
target_positions = np.concatenate([pos_dict['V1_Exc_L2/3'].T, pos_dict['V1_Inh_L2/3'].T])
o_dict = pickle_read("data/antolik/oris.pickle")
target_ori = np.concatenate([np.array(o_dict['V1_Exc_L2/3']), np.array(o_dict['V1_Inh_L2/3'])])

# measure the errors in positions 
s_x = np.zeros((30000))
s_y = np.zeros((30000))
for i in range(30000):
    s_x[i] = abs(x[i] - target_positions[i][0])
    s_y[i] = abs(y[i] + target_positions[i][1])

avg_x = np.sum(s_x)/30000
avg_y = np.sum(s_y)/30000

distances = np.sqrt(s_x**2 + s_y**2)

print(f"The average distance error in x is {avg_x}")
print(f"The average distance error in y is {avg_y}")
print(f"The average distance error is {np.sum(distances)/30000}")

avg_error = np.mean(distances)
distances = [x for x in distances if x < 0.1 * factor] # delete outliers

# and finally plot the distribution of distance errors
plt.gcf().set_size_inches(8, 6.5)
plt.hist(distances, bins=100, color="crimson")
plt.axvline(avg_error, color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(avg_error*1.1, max_ylim*0.9, 'Average error: {:.2f} degrees of visual angle'.format(avg_error))
plt.title("A distribution of errors in predicted neural locations\nwithout outliars ({:.2f} % of the neurons) \n".format((1 - len(distances)/30000) * 100))
plt.xlabel("Error distance [degrees of visual angle]")
plt.ylabel("Number of neurons")
plt.savefig(img_path + "distribution_of_distances_errors", dpi = 300)
plt.clf()



#####################################################################################
############################### Orientation errors ##################################
#####################################################################################

predicted_ori = np.array(ori)
predicted_ori = (2 * np.pi * (predicted_ori + 1) / 2) % np.pi
result1 = rotate_and_compute_distances(predicted_ori, target_ori, period=np.pi, n_rot=36)

predicted_ori = 180 * predicted_ori / np.pi
target_ori = 180 * target_ori / np.pi
result2 = rotate_and_compute_distances(predicted_ori, target_ori, period=180, n_rot=36)


f, ax = plt.subplots()

print("Average orientation error:")
print(result2[0][np.argmin(result2[0])])

print("Shift:")
print(result2[1][np.argmin(result2[0])])
shift = result2[1][np.argmin(result2[0])]

distances = circular_distance(((predicted_ori + shift) % 180), target_ori, 180) 

ori = (predicted_ori + shift) % 180
visualize(12, 2.4, 2.4, "_antolik")

ori = target_ori

x = target_positions[:, 0]
y = target_positions[:, 1]
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
plt.savefig(img_path + "distribution_of_orientation_errors", dpi = 300)
plt.clf()


