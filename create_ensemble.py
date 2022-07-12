import wandb
from model_trainer import (
    Lurz_dataset_preparation_function,
    Antolik_dataset_preparation_function_test,
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

config = {
    "lr": 0.001,
    "test_average_batch": False,
    "compute_oracle_fraction": False,
    "conservative_oracle": True,
    "jackknife_oracle": True,
    "generate_oracle_figure": False,
    "batch_size": 10,
}

run = wandb.init(project="reCNN_visual_prosthesis", entity="csng-cuni")

# You have to edit this by hand. Unfortunatelly, there is no other way around in wandb yet.
artifact_names = [
    "csng-cuni/reCNN_visual_prosthesis/model-bzt64a1d:v0",
    "csng-cuni/reCNN_visual_prosthesis/model-ln4w4h4j:v0",
    # "csng-cuni/reCNN_visual_prosthesis/model-bzt64a1d:v0",
    # "csng-cuni/reCNN_visual_prosthesis/model-bzt64a1d:v0",
]

artifact_dir = None

for a_name in artifact_names:
    artifact = run.use_artifact(a_name, type="model")
    artifact_dir = artifact.download()

print(artifact_dir)

models_paths_list = glob.glob(artifact_dir + "/*.ckpt")
list_of_models = []
for path in models_paths_list:
    m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(path)
    m.freeze()
    list_of_models.append(m)
    print(f"Model from {path} loaded!")


class AggregateModel(ExtendedEncodingModel):
    def __init__(self, models_list, config):
        super().__init__(**config)
        self.models = nn.ModuleList(models_list)

    def forward(self, x):
        x = torch.mean(torch.stack([m(x) for m in self.models], -1), dim=-1)
        return x


model = AggregateModel(list_of_models, config)

# dm = Lurz_dataset_preparation_function(config, None)
dm = Antolik_dataset_preparation_function_test(config, None)

wandb_logger = WandbLogger(log_model=True)
trainer = pl.Trainer(gpus=[0], logger=wandb_logger)
dm.model_performances(m, trainer)
