import wandb
import glob
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d
import matplotlib.pyplot as plt
import pickle
import numpy as np

plt.rcParams.update({'font.size': 12})

def pickle_read(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x

def st(x):
    print(np.min(x))
    print(np.max(x))
    print("------------")


pos_dict = pickle_read("data/antolik/position_dictionary.pickle")
positions = np.concatenate([pos_dict['V1_Exc_L2/3'].T, pos_dict['V1_Inh_L2/3'].T])
o_dict = pickle_read("data/antolik/oris.pickle")
ori = np.concatenate([np.array(o_dict['V1_Exc_L2/3']), np.array(o_dict['V1_Inh_L2/3'])])

run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")
pl.seed_everything(42)

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
f, ax = plt.subplots()
xtrue = positions[:,0]
ytrue = positions[:,1]

positions_copy = positions

factor = 5.5
positions = positions_copy/factor
s_x = np.zeros((30000))
s_y = np.zeros((30000))
for i in range(30000):
    s_x[i] = abs(x[i] - positions[i][0])
    s_y[i] = abs(y[i] + positions[i][1])

avg_x = np.sum(s_x)/30000
avg_y = np.sum(s_y)/30000

distances = np.sqrt(s_x**2 + s_y**2)

print("----------")
print(f"The average distance in x is {avg_x}")
print(f"The average distance in y is {avg_y}")
print(f"The average distance is {np.sum(distances)/30000}")


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


pred = np.array(z)
pred = (2 * np.pi * (pred + 1) / 2) % np.pi
target = ori
result1 = rotate_and_cumpute_distances(pred, target, period=np.pi, n_rot=36)

pred = 180 * pred / np.pi
target = 180 * target / np.pi
result2 = rotate_and_cumpute_distances(pred, target, period=180, n_rot=36)

print("Average error:")
print(result2[0][np.argmin(result2[0])])

print("Shift:")
print(result2[1][np.argmin(result2[0])])

distances *= 5.5

avg_error = np.mean(distances)
distances = [x for x in distances if x < 0.1 * 5.5]

plt.gcf().set_size_inches(8, 6.5)
plt.hist(distances, bins=100, color="crimson")
plt.axvline(avg_error, color='k', linestyle='dashed', linewidth=1)

min_ylim, max_ylim = plt.ylim()
plt.text(avg_error*1.1, max_ylim*0.9, 'Average error: {:.2f} degrees of visual angle'.format(avg_error))
plt.title("A distribution of errors in predicted neural locations\nwithout outliars ({:.2f} % of the neurons) \n".format((1 - len(distances)/30000) * 100))
plt.xlabel("Error distance [degrees of visual angle]")
plt.ylabel("Number of neurons")
plt.savefig("distribution_of_distances_errors", dpi = 300)
plt.clf()