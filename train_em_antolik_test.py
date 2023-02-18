from model_trainer import run_wandb_training
from energy_model.energy_model import EnergyModel
from model_trainer import Antolik_dataset_preparation_function_general
from train_on_antolik import get_config


ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"

model = None

# MODEL
needs_ground_truth = True
model_needs_dataloader = False
init_to_ground_truth_positions = True
init_to_ground_truth_orientations = True
do_not_sample = True # POZOR

# TRAINING
train_on_test = False
max_epochs = 50
max_time = 3
patience = 12
train_on_val = False
test = True
seed = 42
batch_size = 500
lr = 0.01

# DATASET
# brain_crop = 0.5 #None # 0.8
brain_crop = None
stimulus_crop = None # "auto" or [110, 110]

# only on of these can be True (or neither of them), if both were, it would be a double negation and reCNN would be set to the same y as the ground truth 
negative_y_coordinate = True # this says that readout has to put minus in front of y
minus_y = False # this says that dataset has to put minus in front of y when loading the ground truth
# minus_x = True

default_ori_shift = 90
counter_clockwise_rotation = True

    
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
        default_ori_shift,
        counter_clockwise_rotation,
    )

    model = run_wandb_training(
        config,
        Antolik_dataset_preparation_function_general,
        ENTITY,
        PROJECT,
        model_class=EnergyModel
    )
    return model


if __name__ == "__main__":
    model = main()