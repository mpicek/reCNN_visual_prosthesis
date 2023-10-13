import wandb
from model_trainer import (
    Lurz_dataset_preparation_function,
    run_wandb_training,
)
from models import ExtendedEncodingModel, encoding_model
import wandb
import glob
import torch.nn as nn
import torch
from Lurz_dataset import LurzDataModule
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d
import numpy as np
from pytorch_lightning.loggers import WandbLogger
run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")

pl.seed_everything(42)

artifact = run.use_artifact(
    "csng-cuni/reCNN_visual_prosthesis/model-3f2f785r:v0", type="model"
)



config = {
    # GENERAL
    "seed": 24,
    "data_name": "reCNNModel",
    "batch_size": 10,
    "lr": 0.001,
    "max_epochs": 500,
    # CORE GENERAL CONFIG
    "core_hidden_channels": 32,
    "core_layers": 5,
    "core_input_kern": 3,
    "core_hidden_kern": 7,
    # ROTATION EQUIVARIANCE CORE CONFIG
    "num_rotations": 16,
    "stride": 1,
    "upsampling": 1,
    "rot_eq_batch_norm": True,
    "stack": -1,
    "depth_separable": True,
    # READOUT CONFIG
    "readout_bias": True,
    "nonlinearity": "softplus",
    # REGULARIZATION
    "core_gamma_input": 0.019006105312275375,
    "core_gamma_hidden": 0.010224059363842318,
    "readout_gamma": 0.0001391583160413704,
    "input_regularizer": "LaplaceL2norm",  # for RotEqCore - default
    "use_avg_reg": True,
    "reg_readout_spatial_smoothness": 0.0027,
    "reg_group_sparsity": 0.1,
    "reg_spatial_sparsity": 0.45,
    # TRAINER
    "patience": 15,
    "train_on_val": False,  # in case you want to quickly check that your model "compiles" correctly
    "test": True,
    "observed_val_metric": "val/corr",
    "test_average_batch": False,
    # ANTOLIK
    "region": "region1",
    "dataset_artifact_name": "Lurz_dataset:latest",
    "compute_oracle_fraction": True,
    "conservative_oracle": True,
    "jackknife_oracle": True,
    "generate_oracle_figure": True,
}

artifact_dir = artifact.download()

print(artifact_dir)

models_paths_list = glob.glob(artifact_dir + "/*.ckpt")
m = None
for path in models_paths_list:
    m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(path)
    m.freeze()
    print(f"Model from {path} loaded!")

dm = Lurz_dataset_preparation_function(config, None)

wandb_logger = WandbLogger(log_model=True)
trainer = pl.Trainer(gpus=[0],logger=wandb_logger)
dm.model_performances(m, trainer)

# from tqdm import tqdm
# from neuralpredictors.measures.np_functions import oracle_corr_jackknife

# responses = []

# for t in tqdm(dm.get_oracle_dataloader()):
#     img, resp = t
#     # img = img.to("cuda")
#     # resp = resp.to("cuda")
#     if resp.shape[0] != 9:
#         responses.append(resp.detach().cpu().numpy())
#     # img = img[0].unsqueeze(0)
#     # resp = resp.mean(0).unsqueeze(0)
#     print(img.shape)
#     print(resp.shape)
#     # pred = m(img)
#     # print(pred.shape)

# responses = np.stack(responses)
# print(responses)
# print(responses.shape) # (100,)

# out = oracle_corr_jackknife(responses)
# print(out)
# print(out.shape)

