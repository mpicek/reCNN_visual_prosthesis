from model_trainer import run_wandb_training, run_training_without_logging
from models import Lurz_Control_Model
from model_trainer import Antolik_dataset_preparation_function
from train_on_antolik import get_config

ENTITY = "csng-cuni"
PROJECT = "reCNN_visual_prosthesis"
model = None


# MODEL
needs_ground_truth = False
model_needs_dataloader = True
init_to_ground_truth_positions = True
init_to_ground_truth_orientations = False
do_not_sample = False # POZOR

# TRAINING
train_on_test = False
max_epochs = 180
max_time = 1
patience = 5
train_on_val = False
test = True
seed = 42
batch_size = 100
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
        readout_bias=True
    )

    config["train_data_dir"] = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/one_trials.pickle"
    config["test_data_dir"] = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized/ten_trials.pickle"

    model = run_wandb_training(
        config,
        Antolik_dataset_preparation_function,
        ENTITY,
        PROJECT,
        model_class=Lurz_Control_Model
    )
    return model


if __name__ == "__main__":
    model = main()
