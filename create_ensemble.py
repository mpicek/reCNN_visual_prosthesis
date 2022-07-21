import wandb
from model_trainer import (
    Lurz_dataset_preparation_function,
    Antolik_dataset_preparation_function_test,
    run_wandb_training,
)
from models import ExtendedEncodingModel
import wandb
import glob
import torch.nn as nn
import torch
import pytorch_lightning as pl
from models import reCNN_bottleneck_CyclicGauss3d, Lurz_Baseline
from pytorch_lightning.loggers import WandbLogger



class AggregateModel(ExtendedEncodingModel):
    def __init__(self, models_list, config):
        super().__init__(**config)
        self.models = nn.ModuleList(models_list)

    def forward(self, x):
        x = torch.mean(torch.stack([m(x) for m in self.models], -1), dim=-1)
        return x

def download_model(model_name, run):
    """
        Downloads a model from Weights & Biases and returns it.
    """
    artifact = run.use_artifact(model_name, type="model")
    artifact_dir = artifact.download()

    print(artifact_dir)

    model_checkpoint_path = glob.glob(artifact_dir + "/*.ckpt")[0]
    print(model_checkpoint_path)

    m = reCNN_bottleneck_CyclicGauss3d.load_from_checkpoint(model_checkpoint_path)

    print(f"Model from {model_checkpoint_path} loaded!")

    return m

def download_control_model(model_name, run):
    """
        Downloads a model from Weights & Biases and returns it.
    """
    artifact = run.use_artifact(model_name, type="model")
    artifact_dir = artifact.download()

    print(artifact_dir)

    model_checkpoint_path = glob.glob(artifact_dir + "/*.ckpt")[0]
    print(model_checkpoint_path)

    m = Lurz_Baseline.load_from_checkpoint(model_checkpoint_path)

    print(f"Model from {model_checkpoint_path} loaded!")

    return m

def create_ensemble(model_names, config, run, model_name_prefix="", model_name_suffix=""):
    """
        Creates an ensemble given names of the models.
        The ensemble model is returned
    """

    models = []

    for m_name in model_names:
        m = download_model(model_name_prefix + m_name + model_name_suffix, run)
        models.append(m)

    return AggregateModel(models, config)


if __name__ == "__main__":

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
    model_names = [
        "csng-cuni/reCNN_visual_prosthesis/model-uwl9k6yy:v0",
        "csng-cuni/reCNN_visual_prosthesis/model-cj0yrapk:v0",
        "csng-cuni/reCNN_visual_prosthesis/model-7w18n5pj:v0",
        # "csng-cuni/reCNN_visual_prosthesis/model-bzt64a1d:v0",
        # "csng-cuni/reCNN_visual_prosthesis/model-bzt64a1d:v0",
    ]

    model = create_ensemble(model_names)

    dm = Lurz_dataset_preparation_function(config, None)

    wandb_logger = WandbLogger(log_model=True)
    trainer = pl.Trainer(gpus=[0], logger=wandb_logger)
    dm.model_performances(model, trainer)
