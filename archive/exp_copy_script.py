#%%
import torch
import numpy as np

from experiments.utils import pickle_read
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import make_grid

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
import pytorch_lightning as pl
from neuralpredictors.measures.modules import Corr, PoissonLoss
from torch.nn import Parameter
from energy_model.lucas_gabor_filter import GaborFilter
from energy_model.utils import plot_f, create_grating
from energy_model.energy_model import EnergyModel
from datetime import timedelta



ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"
ground_truth_positions_file_path = "data/antolik/position_dictionary.pickle"
ground_truth_orientations_file_path = "data/antolik/oris.pickle"

model = None

config = {
    # GENERAL
    "seed": 2,
    "batch_size": 10,
    "lr": 0.01,
    "max_epochs": 100,

    # CORE GENERAL CONFIG
    "core_hidden_channels": 8,
    "core_layers": 5,
    "core_input_kern": 7,
    "core_hidden_kern": 9,

    # ROTATION EQUIVARIANCE CORE CONFIG
    "num_rotations": 8,       
    "stride": 1,               
    "upsampling": 2,           
    "rot_eq_batch_norm": True, 
    "stack": -1 ,               
    "depth_separable": True,

    # READOUT CONFIG
    "readout_bias": False,
    "nonlinearity": "softplus",
    
    # REGULARIZATION
    "core_gamma_input": 0.00307424496692959,
    "core_gamma_hidden": 0.28463619129195233,
    "readout_gamma": 0.17,
    "input_regularizer": "LaplaceL2norm", # for RotEqCore - default 
    "use_avg_reg": True,

    "reg_readout_spatial_smoothness": 0.0027,
    "reg_group_sparsity": 0.1,
    "reg_spatial_sparsity": 0.45,

    # TRAINER
    "patience": 7,
    "train_on_val": False, # in case you want to quickly check that your model "compiles" correctly
    "test": True,
    "observed_val_metric": "val/corr",

    "test_average_batch": False,
    "compute_oracle_fraction": False,
    "conservative_oracle": True,
    "jackknife_oracle": True,
    "generate_oracle_figure": False,

    # ANTOLIK
    "region": "region1",
    "dataset_artifact_name": "Antolik_dataset:latest",

    # BOTTLENECK
    "bottleneck_kernel": 15,

    "fixed_sigma": False,
    "init_mu_range": 0.9,
    "init_sigma_range": 0.8,

}

# from Antolik_dataset import AntolikDataModule

path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/one_trials.pickle"
path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/ten_trials.pickle"

dataset_config = {
    "train_data_dir": path_train,
    "test_data_dir": path_test,
    "batch_size": config["batch_size"],
    "normalize": True,
    "val_size": 500,
    "brain_crop": None,
    "stimulus_crop": None,
    # "brain_crop": 0.8,
    # "stimulus_crop": "auto",
    # "stimulus_crop": [110, 110],
    # "ground_truth_positions_file_path": "data/antolik/position_dictionary.pickle",
}


import pickle
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from neuralpredictors.data.samplers import SubsetSequentialSampler
from typing import Optional
import pathlib
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from experiments.utils import pickle_read
from Antolik_dataset import AntolikDataModule

dm = AntolikDataModule(**dataset_config)

dm.prepare_data()
dm.setup()

config.update(
        {
            "input_channels": dm.get_input_shape()[0],
            "input_size_x": dm.get_input_shape()[1],
            "input_size_y": dm.get_input_shape()[2],
            "num_neurons": dm.get_output_shape()[0],
            "mean_activity": dm.get_mean(),
            "filtered_neurons":dm.get_filtered_neurons(),
        }
    )

config.update(
    {
        # "ground_truth_positions_file_path": "data/antolik/position_dictionary.pickle",
        # "ground_truth_orientations_file_path": "data/antolik/oris.pickle",
        "ground_truth_positions_file_path": "data/antolik/positions_reparametrized.pickle",
        "ground_truth_orientations_file_path": "data/antolik/oris_reparametrized.pickle",
        "init_to_ground_truth_positions": False,
        "init_to_ground_truth_orientations": False,
        "freeze_positions": False,
        "freeze_orientations": False,
        "orientation_shift": 87.4,
        "factor": 5.5,
        "sample": False,
        "filtered_neurons":None,
    }
)



early_stopping_monitor="val/corr"
early_stopping_mode="max"
model_checkpoint_monitor="val/corr"
model_checkpoint_mode="max"

use_wandb = True


from models import reCNN_bottleneck_CyclicGauss3d_no_scaling

config["positions_minus_x"] = False
config["positions_minus_y"] = True
config["do_not_sample"] = True

model_artifact_name = None
needs_ground_truth = False
model_needs_dataloader = True
model_class = reCNN_bottleneck_CyclicGauss3d_no_scaling
if needs_ground_truth:
    pos_x, pos_y, orientations = dm.get_ground_truth(config["ground_truth_positions_file_path"], config["ground_truth_orientations_file_path"])
    resolution = (dm.get_input_shape()[1], dm.get_input_shape()[2])
    xlim = [-dm.get_stimulus_visual_angle()/2, dm.get_stimulus_visual_angle()/2]
    ylim = [-dm.get_stimulus_visual_angle()/2, dm.get_stimulus_visual_angle()/2]
    # model = model_class(pos_x, pos_y, orientations, resolution, xlim, ylim, **config)
elif model_needs_dataloader:
    model = model_class(dm, **config)
else:
    model = model_class(**config)


trainer = pl.Trainer(
    callbacks=[],
    max_epochs=config["max_epochs"],
    gpus=[0],
    logger=False,
    log_every_n_steps=100,
    # deterministic=True,
    enable_checkpointing=True,
    # fast_dev_run=True,
    # fast_dev_run=7
    # limit_train_batches=1
)

trainer.fit(
    model,
    train_dataloaders=dm.val_dataloader(),
    val_dataloaders=dm.val_dataloader(),
)
# %%
