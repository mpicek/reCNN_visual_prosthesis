import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_trainer import run_wandb_training
from energy_model.energy_model import EnergyModel
from model_trainer import Antolik_dataset_preparation_function
from utils import get_config
import random

ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"
model = None


# TRAINING
train_on_test = False
max_epochs = 1
max_time = 1
patience = 5
train_on_val = False
test = True
seed = random.randint(0, 10000)
batch_size = 10
lr = 0.001

def main():

    config = get_config(model="EM")

    # TRAINING PARAMETERS
    config["train_on_test"] = train_on_test
    config["max_epochs"] = max_epochs
    config["max_time"] = max_time
    config["patience"] = patience
    config["train_on_val"] = train_on_val
    config["test"] = test
    config["seed"] = seed
    config["batch_size"] = batch_size
    config["lr"] = lr

    config["train_data_dir"] = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/one_trials.pickle"
    config["test_data_dir"] = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/ten_trials.pickle"

    model = run_wandb_training(
        config,
        Antolik_dataset_preparation_function,
        ENTITY,
        PROJECT,
        model_class=EnergyModel
    )
    return model

if __name__ == "__main__":
    model = main()