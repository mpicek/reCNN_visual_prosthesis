from model_trainer import run_wandb_training
from models import reCNN_bottleneck_CyclicGauss3d_no_scaling, reCNN_bottleneck_CyclicGauss3d
from model_trainer import Antolik_dataset_preparation_function


ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"
model = None

##################################################################
# MODEL
needs_ground_truth = False # does not go to get_config
model_needs_dataloader = False # does not go to get_config
init_to_ground_truth_positions = False
init_to_ground_truth_orientations = False
do_not_sample = False

# TRAINING
train_on_test = False
max_epochs = 500
max_time = 4
patience = 7
train_on_val = False
test = True
seed = 42
batch_size = 15
lr = 0.001

# DATASET
brain_crop = None # 0.8
stimulus_crop = None # "auto" or [110, 110]

# only on of these can be True (or neither of them), if both were, it would be a double negation and reCNN would be set to the same y as the ground truth 
positions_minus_y = False # this says that dataset has to put minus in front of y when loading the ground truth
positions_minus_x = False

################################################################

def get_config(
    needs_ground_truth = False,
    model_needs_dataloader = False,
    init_to_ground_truth_positions = False,
    init_to_ground_truth_orientations = False,
    do_not_sample = False,
    train_on_test = False,
    max_epochs = 500,
    max_time = 4,
    patience = 7,
    train_on_val = False,
    test = True,
    seed = 42,
    batch_size = 15,
    lr = 0.001,
    brain_crop = None,
    stimulus_crop = None,
    default_ori_shift = 90, # for energy model
    counter_clockwise_rotation = True, # for energy model
    positions_minus_y = True,
    positions_minus_x = False,
    positions_swap_axes = False,
    readout_bias = False,
):
    # path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/one_trials.pickle"
    # path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/ten_trials.pickle"

    import os
    scratch = os.getenv("SCRATCHDIR")
    path_train = scratch + "/one_trials.pickle"
    path_test = scratch + "/ten_trials.pickle"


    # path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/one_trials.pickle"
    # path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/ten_trials.pickle"

    config = {
        # GENERAL
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "max_epochs": max_epochs,
        "needs_ground_truth": needs_ground_truth,
        "model_needs_dataloader": model_needs_dataloader,
        # CORE GENERAL CONFIG
        "core_hidden_channels": 3,
        # "core_layers": 3,
        "core_layers": 1,
        "core_input_kern": 3,
        "core_hidden_kern": 3,
        # ROTATION EQUIVARIANCE CORE CONFIG
        # "num_rotations": 8,
        "num_rotations": 4,
        "stride": 1,
        "upsampling": 2,
        "rot_eq_batch_norm": True,
        "stack": -1,
        "depth_separable": True,
        # READOUT CONFIG
        "readout_bias": readout_bias,
        "nonlinearity": "softplus",
        "do_not_sample": do_not_sample,
        # REGULARIZATION
        "core_gamma_input": 0.00307424496692959,
        "core_gamma_hidden": 0.28463619129195233,
        "readout_gamma": 0.17,
        "input_regularizer": "LaplaceL2norm",  # for RotEqCore - default
        "use_avg_reg": True,
        "reg_readout_spatial_smoothness": 0.0027,
        "reg_group_sparsity": 0.1,
        "reg_spatial_sparsity": 0.45,
        # ANTOLIK
        "dataset_artifact_name": "Antolik_dataset:latest",
        # BOTTLENECK
        "bottleneck_kernel": 15,
        "fixed_sigma": False,
        "init_mu_range": 0.3,
        "init_sigma_range": 0.1,

        # TRAINER
        "patience": patience,
        "train_on_val": train_on_val,  # in case you want to quickly check that your model "compiles" correctly
        "test": test,
        "observed_val_metric": "val/corr",
        "test_average_batch": False,
        "compute_oracle_fraction": False,
        "conservative_oracle": True,
        "jackknife_oracle": True,
        "generate_oracle_figure": False,
    }

    if max_time > 0:
        config.update(
            {
                "max_time": max_time,
            }
        )

    # THE NEW REPARAMETRIZED ANTOLIK DATASET CONFIG
    # CONFIG FOR THE TYPE OF THE MODEL
    #    - init to ground truth positions/orientations
    #    - freeze those positions/orientations
    #    - ori_shift, factor, sample, filtered neurons
    config.update(
        {
            "ground_truth_positions_file_path": "data/antolik/positions_reparametrized.pickle", # "data/antolik/position_dictionary.pickle".. old
            "ground_truth_orientations_file_path": "data/antolik/oris_reparametrized.pickle",
            "init_to_ground_truth_positions": init_to_ground_truth_positions,
            "init_to_ground_truth_orientations": init_to_ground_truth_orientations,
            "freeze_positions": False,
            "freeze_orientations": False,
            "orientation_shift": 87.4,
            "factor": 5.5,
            "sample": False,
        }
    )

    config.update(
        {
            "train_data_dir": path_train,
            "test_data_dir": path_test,
            "normalize": True,
            "val_size": 5000,
            "brain_crop": brain_crop,
            "stimulus_crop": stimulus_crop,
            "positions_minus_y": positions_minus_y,
            "positions_minus_x": positions_minus_x,
            "positions_swap_axes": positions_swap_axes,
        }
    )

    config.update({"train_on_test": train_on_test})

    # for energy model:
    config.update(
        {
            "default_ori_shift": default_ori_shift,
            "counter_clockwise_rotation": counter_clockwise_rotation,
        }
    )

    return config



def main():

    config = get_config(
        needs_ground_truth,
        model_needs_dataloader,
        init_to_ground_truth_positions,
        init_to_ground_truth_orientations,
        do_not_sample,
        train_on_test,
        max_epochs,
        max_time,
        patience,
        train_on_val,
        test,
        seed,
        batch_size,
        lr,
        brain_crop,
        stimulus_crop,
        positions_minus_y,
    )

    model = run_wandb_training(
        config,
        Antolik_dataset_preparation_function,
        ENTITY,
        PROJECT,
        model_class=reCNN_bottleneck_CyclicGauss3d_no_scaling
    )
    return model


if __name__ == "__main__":
    model = main()
