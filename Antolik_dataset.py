import pickle
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from neuralpredictors.data.samplers import SubsetSequentialSampler
from typing import Optional
import pathlib
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from experiments.utils import pickle_read, download_model, reconstruct_orientation_maps, visualize_preferred_orientations



class AntolikDataset(Dataset):
    """A class for handling with the Antolik's synthetic dataset."""    

    def __init__(self, path, normalize=True, brain_crop=None, stimulus_crop=None, ground_truth_path="data/antolik/position_dictionary.pickle"):
        """The constructor.

        Args:
            path (str): Path to the dataset
            normalize (bool, optional): Whether to normalize the images. Defaults to True.
            brain_crop (float, optional): How much of the neurons to take into account (center crop a given area of the brain)
                - None if using all the neurons, 
                - (kept neurons in vis angle in range [0, 2]) if some neurons are cropped out (out of 2)
                - determined by AntolikDataModule
            stimulus_crop (tuple, optional): Whether to center crop the image.
                - None if no crop applied (default), otherwise (kept height pixels, kept width pixels) if cropped.
                - determined by AntolikDataModule
            ground_truth_path (str, optional): Path to the file with the ground truth of positions of neurons.
        """
        self.normalize = normalize
        self.brain_crop = brain_crop
        self.stimulus_crop = stimulus_crop
        self.ground_truth_path = ground_truth_path
        self.filtered = None
        
        if self.stimulus_crop:
            self.set_stimulus_crop(self.stimulus_crop)
            

        self.data = self.pickle_read(path)

        self.transform_list = transforms.Compose(
            # [transforms.Normalize((45.2315,), (26.6845,))] # old dataset
            [transforms.Normalize((46.25135729356395,), (26.337162920481937,))] # new dataset
        )

        if self.brain_crop:
            self.set_brain_crop(self.brain_crop)


    def set_stimulus_crop(self, stimulus_crop):
        self.stimulus_crop = stimulus_crop
        self.crop_transform = transforms.CenterCrop(self.stimulus_crop)
    
    def set_brain_crop(self, brain_crop):
        self.brain_crop = brain_crop
        pos_dict = self.pickle_read(self.ground_truth_path)
        target_positions = np.concatenate([pos_dict['V1_Exc_L2/3'].T, pos_dict['V1_Inh_L2/3'].T])
        self.filtered = np.where((np.abs(target_positions[:, 0]) <= self.brain_crop) & (np.abs(target_positions[:, 1]) <= self.brain_crop))[0]
    
    def get_filtered_neurons(self):
        """Returns filtered neurons if brain crop applied, otherwise returns None
        """
        return self.filtered

    def __getitem__(self, index):
        """Gets the index-th pair of visual stimulus and response to the stimulus.

        Args:
            index (int): the index

        Returns:
            (np.array, np.array): The index-th pair of visual stimulus and response to the stimulus.
        """
        x = self.data[index]["stimulus"]
        x = np.expand_dims(x, axis=0)
        y = np.concatenate(
            [self.data[index]["V1_Exc_L2/3"], self.data[index]["V1_Inh_L2/3"]]
        )

        if self.brain_crop:
            y = y[self.filtered]

        data = torch.from_numpy(x)
        target = torch.from_numpy(y)

        if self.normalize:
            data = self.transform_list(data)
        
        if self.stimulus_crop:
            data = self.crop_transform(data)

        return (data.float(), target.float())

    
    def visualize(self, index):
        stimulus, _ = self.__getitem__(index)
        plt.imshow(stimulus.numpy()[0], cmap='gray')

    def __len__(self):
        """

        Returns:
            int: The length of the dataset
        """
        return len(self.data)

    def pickle_read(self, path):
        """A helper function to unpickle the dataset.

        Args:
            path (str): Path to the dataset

        Returns:
            The dataset.
        """
        with open(path, "rb") as f:
            x = pickle.load(f)
        return x
    
    def get_indices(self):
        """
        Indices of the dataset are not trivial (not integers from 0 to n).
        Each index is a complex string characterising the stimulus-response pair.

        This method returns a list of indices so that we can get a stimulus-response
        pair by typing dataset[indices[i]].
        """
        return [list(self.data.keys())[i] for i in range(self.__len__())]
    
    def get_ground_truth(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, in_degrees=False, minus_y=False, minus_x=False, swap_axes=False, **kwargs):
        """Returns positions in x and y dimensions (in degrees of visual angle) and preferred orientations
           (in radians) of the Antolik's model's ground truth.
        

        Args:
            ground_truth_positions_file_path (str): Path to the file with positions of neurons
            ground_truth_orientations_file_path (str): Path to the file with preferred orientations of neurons
            in_degrees (Bool, optional): If we want to return the orientations in degrees. If
                False, orientations in radians are returned. Defaults to False.

        Returns:
            tuple: numpy arrays pos_x, pos_y, target_ori (in radians!)
        """
        filtered_neurons = self.get_filtered_neurons()

        pos_dict = pickle_read(ground_truth_positions_file_path)
        target_positions = np.concatenate([pos_dict['V1_Exc_L2/3'].T, pos_dict['V1_Inh_L2/3'].T])

        if filtered_neurons is not None:
            target_positions = target_positions[filtered_neurons, :]

        pos_x = None
        if minus_x:
            pos_x = (-target_positions[:,0])
        else:
            pos_x = (target_positions[:,0])

        pos_y = None
        if minus_y:
            pos_y = (-target_positions[:,1])
        else:
            pos_y = (target_positions[:,1])
        
        if swap_axes:
            pos_x, pos_y = pos_y, pos_x


        o_dict = pickle_read(ground_truth_orientations_file_path)
        target_ori = np.concatenate([np.array(o_dict['V1_Exc_L2/3']), np.array(o_dict['V1_Inh_L2/3'])])
        
        if in_degrees:
            target_ori = 180*(target_ori / np.pi) # from [0, pi] to [0, 180]

        if filtered_neurons is not None:
            target_ori = target_ori[self.get_filtered_neurons()]

        return pos_x, pos_y, target_ori


class AntolikDataModule(pl.LightningDataModule):
    """
    Warning! Has to be downloaded from the wintermute server!
    A Pytorch Lightning module that uses Antolik's dataset.
    """

    def __init__(
        self,
        train_data_dir,
        test_data_dir,
        batch_size,
        # seed=None,
        normalize=True,
        num_workers=0,
        val_size=5000,
        brain_crop=None,
        stimulus_crop=None,
        ground_truth_path="data/antolik/position_dictionary.pickle",
        original_stimulus_visual_angle=11,
        original_stimulus_resolution=110
    ):
        """The constructor.

        Args:
            train_data_dir (str): Path to the train dataset
            test_data_dir (str): Path to the test dataset
            batch_size (int): Batch size
            normalize (bool, optional): Whether to normalize the input images. Defaults to True.
            num_workers (int, optional): Number of workers that load the dataset. Defaults to 0.
            val_size (int, optional): Validation dataset length. Defaults to 5000.
            brain_crop (float, optional): How much of the neurons to take into account (center crop a given area of the brain)
                - None if using all the neurons, 
                - (kept width neurons in vis angle from interval [0, 2]) if some neurons are cropped out out of 2
            stimulus_crop (tuple, optional): How much to center crop the image.
                - None if no crop applied (default), 
                - "auto" to compute automatically (only if brain_crop defined)
                - otherwise (kept height pixels, kept width pixels) if cropped.
            ground_truth_path (str, optional): path to the .pickle file with dictionary of
                positions of neurons (ground truth from the model)
            original_stimulus_visual_angle (float, optional): How much of visual angle the original uncropped stimulus spans.
                - Default: 11 deg of vis angle, that means 5.5 deg of vis angle to each side, that is 5.5 to the right, 5.5 to the left, up and down
                - this argument might be somewhere in code named x_lim and y_lim (and it is, therefore, for a square image)
            original_stimulus_resolution (int, optional): original resolution of the stimulus (uncropped).. one side, it is a square
        """
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        # seed=None,
        self.normalize = normalize
        self.num_workers = num_workers
        self.val_size = val_size
        self.brain_crop = brain_crop

        self.factor = 5.5 # predefined constant 
        self.stimulus_crop = stimulus_crop


        self.original_stimulus_resolution = original_stimulus_resolution
        self.original_stimulus_visual_angle = original_stimulus_visual_angle
        # it is uncropped, so by default, the stimulus_visual_angle is the same as initialized
        # (but will be adjusted in set_stimulus_crop)
        self.stimulus_visual_angle = original_stimulus_visual_angle


        # automatically compute the crop of the stimulus image
        if self.brain_crop and self.stimulus_crop == "auto":
            keep_visual_field = self.brain_crop # this is how much of visual field (out of 2) is kept (from the middle). The rest is cropped out

            # neurons are from -2 to +2, when normalized to [-1, 1] (where, stimulus is presented to the whole area of [-1, 1]),
            # we want to know how much of the space the neurons allocate. All the following computations are only in one quadrant
            normalized_area_of_neurons = 2/self.factor # ~0.36.. out of 1
            # given a crop of neurons (keep_visual_field), we want to know, how many pixels from the stimulus we can crop

            cropped_neurons = 2-keep_visual_field
            ratio_of_cropped_neurons = cropped_neurons / 2

            normalized_area_of_cropped_neurons = normalized_area_of_neurons * ratio_of_cropped_neurons

            discard_pixels_each_side = math.floor((self.original_stimulus_resolution/2) * normalized_area_of_cropped_neurons)

            print(str(discard_pixels_each_side) + "px will be discarded from each side.")
            self.stimulus_crop = (self.original_stimulus_resolution - 2*discard_pixels_each_side, self.original_stimulus_resolution - 2*discard_pixels_each_side)

            self.stimulus_visual_angle = (self.original_stimulus_visual_angle / self.original_stimulus_resolution) * self.stimulus_crop[0] # it is a square
        
        elif self.stimulus_crop is not None:
            self.stimulus_visual_angle = (self.original_stimulus_visual_angle / self.original_stimulus_resolution) * self.stimulus_crop[0] # it is a square

        self.ground_truth_path = ground_truth_path


    def prepare_data(self):
        """We do not have public access to the data. This function will be implemented
        when the dataset is available on some website.

        Raises:
            Exception: The train .pickle file does not exist
            Exception: The test .pickle file does not exist
        """        

        # we should not do anything like self.x = y # = assign state
        # just download the data
        train_path = pathlib.Path(self.train_data_dir)
        test_path = pathlib.Path(self.test_data_dir)

        if not train_path.exists():
            raise Exception(
                f"File {str(train_path)} with Antolik train dataset does not exist."
            )

        if not test_path.exists():
            raise Exception(
                f"File {str(train_path)} with Antolik test dataset does not exist."
            )

    def setup(self, stage: Optional[str] = None):
        """Sets up the dataset, loads it, shuffles it and sets up the sampler.

        Args:
            stage (Optional[str], optional): Possible values are 'fit, 'test',
            'predict' or None. None means both 'fit' and 'test'. If 'fit', the
            method sets up only the train dataset. If 'test', it sets up only
            the train dataset. If 'predict', it sets up the train dataset.
            Defaults to None.
        """
        # stage is "fit" or "test" or "predict"
        # when stage=None -> both "fit" and "test"

        self.train_dataset = AntolikDataset(
            self.train_data_dir, self.normalize, self.brain_crop, self.stimulus_crop, self.ground_truth_path
        )

        self.test_dataset = AntolikDataset(self.test_data_dir, self.normalize, self.brain_crop, self.stimulus_crop, self.ground_truth_path)

        print("Data loaded successfully!")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage is None:

            indices = np.arange(0, len(self.train_dataset))

            rng = np.random.default_rng(69)
            rng.shuffle(indices)
            indices_keys = [list(self.train_dataset.data.keys())[i] for i in indices]

            subset_idx_val = indices_keys[0 : self.val_size]
            subset_idx_train = indices_keys[self.val_size :]

            self.subset_idx_val = subset_idx_val

            self.train_random_sampler = SubsetRandomSampler(subset_idx_train)
            self.train_sequential_sampler = SubsetSequentialSampler(subset_idx_train)
            self.val_sampler = SubsetSequentialSampler(subset_idx_val)

        if stage == "test" or stage is None:
            indices = np.arange(0, len(self.test_dataset))
            subset_idx_test = [list(self.test_dataset.data.keys())[i] for i in indices]
            self.test_sampler = SubsetSequentialSampler(subset_idx_test)

    def get_stimulus_visual_angle(self):
        """Returns how much the stimulus spans.

        Returns:
            float: How much the stimulus spans. If the stimulus_crop is defined, then returns
                how much this cropped stimulus spans. The whole span is returned, not to one and other side.
                Returns float as the stimulus is a square.
        """
        return self.stimulus_visual_angle
    
    def get_ground_truth(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, in_degrees=False, minus_y=False, minus_x=False, swap_axes=False, **kwargs):
        """Returns positions in x and y dimensions (in degrees of visual angle) and preferred orientations
           (in radians) of the Antolik's model's ground truth.
        

        Args:
            ground_truth_positions_file_path (str): Path to the file with positions of neurons
            ground_truth_orientations_file_path (str): Path to the file with preferred orientations of neurons
            in_degrees (Bool, optional): If we want to return the orientations in degrees. If
                False, orientations in radians are returned. Defaults to False.

        Returns:
            tuple: numpy arrays pos_x, pos_y, target_ori (in radians!)
        """
        return self.train_dataset.get_ground_truth(ground_truth_positions_file_path, ground_truth_orientations_file_path, in_degrees, minus_y, minus_x, swap_axes)
    
    def visualize_orientation_map(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, save=False, img_path="img/", suffix="_truth", neuron_dot_size=5, in_degrees=False, minus_y=False, minus_x=False, swap_axes=False):
        
        fig, ax = plt.subplots()
        x, y, o = self.get_ground_truth(ground_truth_positions_file_path, ground_truth_orientations_file_path, in_degrees, minus_y, minus_x, swap_axes)
        reconstruct_orientation_maps(x, y, o, fig, ax, save, 12, 2.4, 2.4, img_path, suffix, neuron_dot_size)

    def get_input_shape(self):
        x, _ = next(iter(self.train_dataloader()))
        return x[0].shape

    def get_output_shape(self):
        _, y = next(iter(self.train_dataloader()))
        return y[0].shape
    
    def get_filtered_neurons(self):
        return self.train_dataset.get_filtered_neurons()

    def get_mean(self):
        """Computes the mean response of the train dataset. If it is available
        in a locally generated file, it loads it from there. Otherwise it
        computes it and then it stores it into the file for the future use.

        Returns:
            torch.Tensor: Mean responses of the neurons.
        """        

        mean_path = pathlib.Path(
            self.train_data_dir.rsplit(".pickle", 1)[0] + "_mean.npy"
        )

        if mean_path.exists():
            mean = np.load(mean_path)
            print("Loaded precomputed mean from " + str(mean_path))
            return torch.from_numpy(mean)

        dataloader = DataLoader(
            self.train_dataset,
            sampler=self.train_sequential_sampler,
            batch_size=self.batch_size,
            drop_last=True,
        )
        summed = torch.zeros(self.get_output_shape())

        for (_, y) in dataloader:
            summed += torch.sum(y, 0)

        mean = summed / self.train_len()

        np.save(mean_path, mean)
        print("Created mean array and saved to " + str(mean_path))
        return mean

    def train_len(self):
        return len(self.train_random_sampler)

    def val_len(self):
        return len(self.val_sampler)

    def test_len(self):
        return len(self.test_sampler)

    def __len__(self):
        """The length of ALL the data we have (train + val + test)

        Returns:
            int: The length of ALL the data we have (train + val + test)
        """        
        return self.train_len() + self.val_len() + self.test_len()

    def print_dataset_info(self):
        """Creates a train dataloader, gets first piece of data and prints its shape
        """
        print(" ------------ DATASET INFO ------------ ")
        print(" SHAPES:")
        dataloader = DataLoader(
            self.train_dataset,
            sampler=self.train_random_sampler,
            batch_size=self.batch_size,
            drop_last=True,
        )
        print(f"    Input shape (images): {self.get_input_shape()}")
        print("    With batch size also: ", end="")
        print(next(iter(dataloader))[0].shape)

        print(f"    Output shape (responses): {self.get_output_shape()}")
        print("    With batch size also: ", end="")
        print(next(iter(dataloader))[1].shape)

        print(" LENGTH:")
        print(f"    Length of the dataset is {len(self)}")
        print(f"    Length of the train set is {self.train_len()}")
        print(f"    Length of the val set is {self.val_len()}")
        print(f"    Length of the test set is {self.test_len()}")

        print(" -------------------------------------- ")

    def train_dataloader(self):
        """
        Returns:
            DataLoader: The train DataLoader
        """        
        return DataLoader(
            self.train_dataset,
            sampler=self.train_random_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        """
        Validation data are in the variable train_data 
        (but the indices are splitted to self.val_sampler)

        Returns:
            DataLoader: The validation DataLoader
        """   
        return DataLoader(
            self.train_dataset,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        """
        Returns:
            DataLoader: The test DataLoader
        """   
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def get_oracle_dataloader(self):
        """the only difference from test_dataloader is that we hardcode batch_size=10

        Returns:
            DataLoader: The test DataLoader with batch_size=10
        """
        
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=10,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def model_performances(self, model=None, trainer=None, control_measures=None):
        """Evaluates the model and prints the results

        Args:
            model (pl.model, optional): Model to be evaluated. Defaults to None.
            trainer (pl.trainer, optional): The trainer that performs the evaluation. Defaults to None.
            control_measures (dict, optional): The control model's measures to be compared with our evaluated model. Defaults to None.

        Returns:
            dict: Dictionary of the resulting measures.
        """
        model.test_average_batch = False
        model.compute_oracle_fraction = False
        val_score = trainer.test(model, self.val_dataloader(), verbose=False)
        test_score = trainer.test(model, self.test_dataloader(), verbose=False)

        model.test_average_batch = True
        model.compute_oracle_fraction = True
        test_repeats_averaged_score = trainer.test(model, self.get_oracle_dataloader(), verbose=False)

        val_score = val_score[0]
        test_score = test_score[0]
        test_repeats_averaged_score = test_repeats_averaged_score[0]

        print("Validation dataset:")
        print(f"    Correlation: {'{:.4f}'.format(val_score['test/corr'])} {'({:.2f} percent of the control model)'.format(100 * (val_score['test/corr'] / control_measures['val/corr'])) if control_measures else ''}")


        # print("Test dataset:")
        # print(f"    Correlation: {'{:.4f}'.format(test_score['test/corr']) }")

        print("Test dataset with averaged responses of repeated trials:")
        print(f"    Correlation: {'{:.4f}'.format(test_repeats_averaged_score['test/repeated_trials/corr']) } {'({:.2f} percent of the control model)'.format(100 * (test_repeats_averaged_score['test/repeated_trials/corr'] / control_measures['test/repeated_trials/corr'])) if control_measures else ''}")
        print(f"    Fraction oracle conservative: {'{:.4f}'.format(test_repeats_averaged_score['test/fraction_oracle_conservative'])} {'({:.2f} percent of the control model)'.format(100 * (test_repeats_averaged_score['test/fraction_oracle_conservative'] / control_measures['test/fraction_oracle_conservative'])) if control_measures else ''}")
        print(f"    Fraction oracle jackknife: {'{:.4f}'.format(test_repeats_averaged_score['test/fraction_oracle_jackknife'])} {'({:.2f} percent of the control model)'.format(100 * (test_repeats_averaged_score['test/fraction_oracle_jackknife'] / control_measures['test/fraction_oracle_jackknife'])) if control_measures else ''}")

        returned_measures = {
            "val/corr": val_score['test/corr'],
            "test/repeated_trials/corr": test_repeats_averaged_score['test/repeated_trials/corr'],
            "test/fraction_oracle_conservative":test_repeats_averaged_score['test/fraction_oracle_conservative'],
            "test/fraction_oracle_jackknife":test_repeats_averaged_score['test/fraction_oracle_jackknife']
        }

        return returned_measures


    def pickle_read(self, path):
        with open(path, "rb") as f:
            x = pickle.load(f)
        return x


if __name__ == "__main__":

    path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/one_trials.pickle"
    path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/ten_trials.pickle"

    path_small_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik/small_train.pickle"

    dm = AntolikDataModule(path_test, path_test, 10, val_size=500)
    dm.prepare_data()
    dm.setup()
    dm.print_dataset_info()
