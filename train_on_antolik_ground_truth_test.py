from model_trainer import run_wandb_training
from models import reCNN_bottleneck_CyclicGauss3d_no_scaling, reCNN_bottleneck_CyclicGauss3d
from model_trainer import Antolik_dataset_preparation_function
from train_on_antolik2 import get_config

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
train_on_test = True
max_epochs = 2
max_time = 4
patience = 1
train_on_val = True
test = True
seed = 42
batch_size = 10
lr = 0.001

# DATASET
# brain_crop = 0.5 #None # 0.8
brain_crop = None
stimulus_crop = None # "auto" or [110, 110]

# only on of these can be True (or neither of them), if both were, it would be a double negation and reCNN would be set to the same y as the ground truth 
negative_y_coordinate = True # this says that readout has to put minus in front of y
minus_y = False # this says that dataset has to put minus in front of y when loading the ground truth



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
        negative_y_coordinate,
        minus_y,
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
