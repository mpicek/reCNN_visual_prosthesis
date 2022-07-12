from model_trainer import run_wandb_training
from models import reCNN_bottleneck_CyclicGauss3d
from model_trainer import Lurz_dataset_preparation_function


ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"

model = None

config = {
    # GENERAL
    "seed": 42,
    "batch_size": 10,
    "lr": 0.0001,
    "max_epochs": 500,
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
    "stack": -1,
    "depth_separable": True,
    # READOUT CONFIG
    "readout_bias": True,
    "nonlinearity": "softplus",
    # REGULARIZATION
    "core_gamma_input": 0.00307424496692959,
    "core_gamma_hidden": 0.28463619129195233,
    "readout_gamma": 0.17,
    "input_regularizer": "LaplaceL2norm",  # for RotEqCore - default
    "use_avg_reg": True,
    "reg_readout_spatial_smoothness": 0.0027,
    "reg_group_sparsity": 0.1,
    "reg_spatial_sparsity": 0.45,
    # TRAINER
    "patience": 10,
    "train_on_val": False,  # in case you want to quickly check that your model runs correctly
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
    # BOTTLENECK
    "bottleneck_kernel": 15,
    # "fixed_sigma": False,
    "fixed_sigma": True,
    "init_mu_range": 0.5,
    "init_sigma_range": 0.5,
}


def main():

    model = run_wandb_training(
        config,
        Lurz_dataset_preparation_function,
        ENTITY,
        PROJECT,
        model_class=reCNN_bottleneck_CyclicGauss3d,
    )
    return model


if __name__ == "__main__":
    model = main()
