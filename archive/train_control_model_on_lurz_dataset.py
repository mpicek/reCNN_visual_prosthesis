from model_trainer import run_wandb_training
from models import Lurz_Control_Model
from model_trainer import Lurz_dataset_preparation_function


ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"

model = None


config = {
    # GENERAL
    "seed": 24,
    "data_name": "PicekModel",
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
    "stack": -3,               
    "depth_separable": True,   # default ... TODO

    # READOUT CONFIG
    "readout_bias": True,
    "nonlinearity": "softplus",

    # REGULARIZATION
    "core_gamma_input": 0.019006105312275375,
    "core_gamma_hidden": 0.010224059363842318,
    "readout_gamma": 0.0001391583160413704,
    "input_regularizer": "LaplaceL2norm", # for RotEqCore - default 
    "use_avg_reg": True,

    "reg_readout_spatial_smoothness": 0.0027,
    "reg_group_sparsity": 0.1,
    "reg_spatial_sparsity": 0.45,

    # TRAINER
    "patience": 10,
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
    "dataset_artifact_name": "Lurz_dataset:latest",
}


def main():

    model = run_wandb_training(config, Lurz_dataset_preparation_function, ENTITY, PROJECT, model_class=Lurz_Control_Model)
    return model


if __name__ == "__main__":
    model = main()