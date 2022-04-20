from model_trainer import run_wandb_training
from models import Lurz, PicekGauss, Picek, LurzRotEq, LurzGauss
from predict_neural_responses.models import CNN_SxF
from model_trainer import Lurz_dataset_preparation_function, Antolik_dataset_preparation_function


ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"

model = None


config = {
    # GENERAL
    "seed": 42,
    "data_name": "PicekModel",
    "batch_size": 25,
    "lr": 0.001,
    "max_epochs": 500,

    # CORE GENERAL CONFIG
    "core_hidden_channels": 64,
    "core_layers": 5,
    "core_input_kern": 3,
    "core_hidden_kern": 3,

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
    "core_gamma_input": 0.1,
    "core_gamma_hidden": 52,
    "readout_gamma": 0.17,
    "input_regularizer": "LaplaceL2norm", # for RotEqCore - default 
    "use_avg_reg": True,

    "reg_readout_spatial_smoothness": 0.0027,
    "reg_group_sparsity": 0.1,
    "reg_spatial_sparsity": 0.45,

    # TRAINER
    "patience": 7,
    "train_on_val": True, # in case you want to quickly check that your model "compiles" correctly
    "test": True,
    "observed_val_metric": "val/corr",

    # ANTOLIK
    "region": "region1",
}


def main():
    
    # config["seed"] = 42
    # config["data_name"] = "Antolik2016"
    # config["region"] = "region1"
    # config["batch_size"] = 128
    # config["lr"] = 0.0001
    # config["max_epochs"] = 3000
    # config["core_hidden_channels"] = 64
    # config["core_layers"] = 4
    # config["core_input_kern"] = 9
    # config["core_hidden_kern"] = 7
    # config["readout_bias"] = True
    # config["core_gamma_input"] = 0.01
    # config["core_gamma_hidden"] = 0.01
    # config["readout_gamma"] = 1

    #TODO: input_channels = 1???????????

    model = run_wandb_training(config, Lurz_dataset_preparation_function, ENTITY, PROJECT, dataset_artifact_name="Lurz_dataset:latest", model_class=Lurz)
    return model
    

if __name__ == "__main__":
    model = main()
