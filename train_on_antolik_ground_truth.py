from model_trainer import run_wandb_training, run_training_without_logging
from models import reCNN_bottleneck_CyclicGauss3d_no_scaling, reCNN_bottleneck_CyclicGauss3d
from model_trainer import Antolik_dataset_preparation_function
from train_on_antolik import get_config

ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"
model = None


# MODEL
needs_ground_truth = False
model_needs_dataloader = True
init_to_ground_truth_positions = True
init_to_ground_truth_orientations = True
do_not_sample = True # POZOR

# TRAINING
train_on_test = False
max_epochs = 10
max_time = 0
patience = 7
train_on_val = False
test = True
seed = 42
batch_size = 10
lr = 0.001

# DATASET
# brain_crop = 0.5 #None # 0.8
brain_crop = None
stimulus_crop = None # "auto" or [110, 110]
# stimulus_crop = "auto"

# only on of these can be True (or neither of them), if both were, it would be a double negation and reCNN would be set to the same y as the ground truth 
# negative_y_coordinate = True # this says that readout has to put minus in front of y
positions_minus_y = True # this says that dataset has to put minus in front of y when loading the ground truth
positions_minus_x = False # this says that dataset has to put minus in front of x when loading the ground truth



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
        positions_minus_x,
    )

    config["core_gamma_hidden"] = 0.008931320307500908
    config["bottleneck_kernel"] = 15
    config["core_gamma_input"] = 0.2384005754453638
    config["core_hidden_channels"] = 6
    config["core_hidden_kern"] = 19
    config["core_input_kern"] = 5
    config["core_layers"] = 5
    config["depth_separable"] = True
    config["lr"] = 0.0005
    config["num_rotations"] = 8
    config["upsampling"] = 1
    config["train_data_dir"] = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/one_trials.pickle"
    config["test_data_dir"] = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/ten_trials.pickle"

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
