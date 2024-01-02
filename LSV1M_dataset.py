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
from experiments.utils import pickle_read, reconstruct_orientation_maps
import os
import shutil
import torch.nn.functional as F
from skimage.transform import resize

class LSV1MDatasetSingleTrial(Dataset):
    """A class for handling the LSV1M Single Trial synthetic dataset."""    

    def __init__(self, path, normalize=True, ground_truth_file=None, **kwargs):
        """The constructor.

        Args:
            path (str): Path to the dataset
            normalize (bool, optional): Whether to normalize the images. Defaults to True.
            ground_truth_file (str, optional): Path to the file with the ground truth of positions and orientations of neurons.
        """
        self.normalize = normalize
        self.path = path
        self.ground_truth_file = ground_truth_file

        self.input_transform_list = transforms.Compose(
            [transforms.Normalize((46.1058,), (26.8956,))]
        )

        self.input_mean = 46.1324
        self.input_std = 24.4895
        self.target_mean = 3.4808
        self.target_std = 3.1232

        self.indices_representation = self.get_internal_indices_representation()
        self.indices = self.get_indices()

    def __getitem__(self, index):
        """Gets the index-th pair of visual stimulus and response to the stimulus.

        Args:
            index (int): the index

        Returns:
            (np.array, np.array): The index-th pair of visual stimulus and response to the stimulus.
        """
        loaded_files = {}

        index = self.indices_representation[index]

        for file in os.listdir(os.path.join(self.path, index)):
            # load npy file into loaded_files without the .npy extension
            filename = os.path.join(self.path, index, file)
            loaded_files[file[:-4]] = np.load(filename)
        
        x = loaded_files['stimulus']
        x = np.expand_dims(x, axis=0)
        y = np.concatenate(
            [loaded_files["V1_Exc_L23"], loaded_files["V1_Inh_L23"]]
        )
        
        data = torch.from_numpy(x)
        target = torch.from_numpy(y)

        if self.normalize:
            data = self.input_transform_list(data)

        return (data.float(), target.float())
    
    def visualize(self, index):
        stimulus, _ = self.__getitem__(index)
        plt.imshow(stimulus.numpy()[0], cmap='gray')

    def __len__(self):
        """

        Returns:
            int: The length of the dataset
        """

        return len(self.indices)
    
    def get_internal_indices_representation(self):
        """
        Indices of the dataset are not trivial (not integers from 0 to n).
        Each index is a complex string characterising the stimulus-response pair.

        This method returns a list of indices so that we can get a stimulus-response
        pair by typing dataset[indices[i]].
        """
        indices = os.listdir(self.path)
        indices.sort()  # Sort the indices in ascending order
        
        return indices

    def get_indices(self):
        """
        Returns:
            np.array: The indices of the dataset (just 0, 1, 2, ... len(indices))
        """
        return np.arange(len(self.indices_representation))
    
    def get_ground_truth(self, ground_truth_file=None, in_degrees=False, positions_minus_y=False, positions_minus_x=False, positions_swap_axes=False, **kwargs):
        """Returns positions in x and y dimensions (in degrees of visual angle) and preferred orientations
           (in radians) of the LSV1M's model's ground truth.
        

        Args:
            ground_truth_file (str, None): Path to the file with positions and orientations of neurons. If none,
                self.ground_truth_positions_file_path is used. Defaults to None.
            in_degrees (Bool, optional): If we want to return the orientations in degrees. If
                False, orientations in radians are returned. Defaults to False.

        Returns:
            tuple: numpy arrays pos_x, pos_y, target_ori (in radians!)
        """

        if ground_truth_file is None:
            ground_truth_file = self.ground_truth_file

        data = pickle_read(ground_truth_file)
        target_positions_x = np.concatenate([data['V1_Exc_L23']['pos_x'], data['V1_Inh_L23']['pos_x']])
        target_positions_y = np.concatenate([data['V1_Exc_L23']['pos_y'], data['V1_Inh_L23']['pos_y']])

        pos_x = None
        if positions_minus_x:
            pos_x = -target_positions_x
        else:
            pos_x = target_positions_x

        pos_y = None
        if positions_minus_y:
            pos_y = -target_positions_y
        else:
            pos_y = target_positions_y

        if positions_swap_axes:
            pos_x, pos_y = pos_y, pos_x


        target_ori = np.concatenate([np.array(data['V1_Exc_L23']['ori']), np.array(data['V1_Inh_L23']['ori'])])
        
        if in_degrees:
            target_ori = 180*(target_ori / np.pi) # from [0, pi] to [0, 180]

        return pos_x, pos_y, target_ori

class LSV1MDatasetMultiTrial(LSV1MDatasetSingleTrial):
    """A class for handling the LSV1M Multi Trial synthetic dataset."""    

    def __init__(self, path, normalize=True, ground_truth_file=None, average_test_trials=True, **kwargs):
        """The constructor.

        Args:
            path (str): Path to the dataset
            normalize (bool, optional): Whether to normalize the images. Defaults to True.
            ground_truth_file (str, optional): Path to the file with the ground truth of positions and orientations of neurons.
        """
        self._average_test_trials = None
        super().__init__(path, normalize, ground_truth_file, **kwargs)
        self.average_test_trials = average_test_trials

    def __getitem__(self, index):
        """Gets the index-th pair of visual stimulus and response to the stimulus.

        Args:
            index (int): the index

        Returns:
            (np.array, np.array): The index-th pair of visual stimulus and response to the stimulus.
        """

        index = self.indices_representation[index]

        stimulus_path = None
        target = None

        if self.average_test_trials:
            stimulus_path = os.path.join(self.path, index, "stimulus.npy")
            target = np.concatenate(
                [np.load(os.path.join(self.path, index, "V1_Exc_L23_averaged.npy")),
                np.load(os.path.join(self.path, index, "V1_Inh_L23_averaged.npy"))]
            )
        else:
            stimulus_path = os.path.join(os.path.dirname(index), "stimulus.npy")
            target = np.concatenate(
                [np.load(os.path.join(index, "V1_Exc_L23.npy")),
                np.load(os.path.join(index, "V1_Inh_L23.npy"))]
            )
        
        x = np.load(stimulus_path)
        x = np.expand_dims(x, axis=0)

        data = torch.from_numpy(x)
        target = torch.from_numpy(target)

        if self.normalize:
            data = self.input_transform_list(data)
        
        return (data.float(), target.float())
    
    def get_internal_indices_representation(self):
        """
        Indices of the dataset are not trivial (not integers from 0 to n).
        Each index is a complex string characterizing the stimulus-response pair.

        This method returns a list of indices so that we can get a stimulus-response
        pair by typing dataset[indices[i]].

        For averaged, the indices are just the number of trials (0000200000, 0000200001, ...),
        for not averaged, it is the whole path for each subtrial ('/some_path/multi_trial/0000200000/trial=0', '/some_path/multi_trial/0000200000/trial=1', ...)
        """

        indices = []

        subfolders = os.listdir(self.path)
        subfolders.sort()  # Sort the indices in ascending order

        if self.average_test_trials:
            for subfolder in subfolders:
                indices.append(subfolder)
        else:
            for subfolder in subfolders:
                trials = os.listdir(os.path.join(self.path, subfolder))
                trials.sort()
                # delete trial with name stimulus.npy, V1_Inh_L23_averaged.npy and V1_Exc_L23_averaged.npy and make the indices whole paths
                whole_path_trials = [os.path.join(self.path, subfolder, trial) for trial in trials if trial != "stimulus.npy" and trial != "V1_Inh_L23_averaged.npy" and trial != "V1_Exc_L23_averaged.npy"]
                indices += whole_path_trials
        
        return indices
    
    def get_indices(self):
        """
        Returns:
            np.array: The indices of the dataset (just 0, 1, 2, ... len(indices))
        """
        return np.arange(len(self.get_internal_indices_representation()))
    
    @property
    def average_test_trials(self):
        return self._average_test_trials
    
    @average_test_trials.setter
    def average_test_trials(self, average_test_trials):
        """
        When we change the average_test_trials, we have to change the indices_representation and indices.
        """
        self._average_test_trials = average_test_trials
        self.indices_representation = self.get_internal_indices_representation()
        self.indices = self.get_indices()
    
    @average_test_trials.getter
    def average_test_trials(self):
        return self._average_test_trials
    
    @average_test_trials.deleter
    def average_test_trials(self):
        del self._average_test_trials



class LSV1MDataModule(pl.LightningDataModule):
    """
    Warning! Has to be downloaded from the gin server!
    A Pytorch Lightning module that uses LSV1M dataset.
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
        average_test_trials=True,
        brain_crop=None,
        stimulus_crop=None,
        ground_truth_file=None,
        original_stimulus_visual_angle=11,
        original_stimulus_resolution=110,
        use_scratch=False,
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
            ground_truth_file (str, optional): path to the .pickle file with dictionary of neuron positions and orientations
            original_stimulus_visual_angle (float, optional): How much of visual angle the original uncropped stimulus spans.
                - Default: 11 deg of vis angle, that means 5.5 deg of vis angle to each side, that is 5.5 to the right, 5.5 to the left, up and down
                - this argument might be somewhere in code named x_lim and y_lim (and it is, therefore, for a square image)
            original_stimulus_resolution (int, optional): original resolution of the stimulus (uncropped).. one side, it is a square
            use_scratch (bool, optional): Whether to copy and extract the dataset (in .zip format) in the scratch directory. Defaults to False.
        """
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        # seed=None,
        self.normalize = normalize
        self.average_test_trials = average_test_trials
        self.num_workers = num_workers
        self.val_size = val_size

        self.factor = 5.5 # predefined constant
        self.stimulus_visual_angle = original_stimulus_visual_angle

        # it is uncropped, so by default, the stimulus_visual_angle is the same as initialized
        # (but will be adjusted in set_stimulus_crop)
        self.ground_truth_file = ground_truth_file

        self.use_scratch = use_scratch

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
                f"File {str(train_path)} with LSV1M train dataset does not exist."
            )

        if not test_path.exists():
            raise Exception(
                f"File {str(train_path)} with LSV1M test dataset does not exist."
            )

        if self.use_scratch:
            self.scratch_path = os.environ.get('SCRATCH')
            if self.scratch_path is None:
                raise Exception("You are trying to use the scratch directory, but the SCRATCH environment variable is not set.")
            
            # get only the filename (for example "single_trial.zip")
            train_filename = os.path.basename(self.train_data_dir)
            test_filename = os.path.basename(self.test_data_dir)

            # the extracted path will have the name of the file but without .zip extension
            # so for example "single_trial" (when the .zip archive is named "single_trial.zip")
            if not pathlib.Path(os.path.join(self.scratch_path, train_filename.split('.')[-2])).exists():
                destination_train = os.path.join(self.scratch_path, train_filename)
                print("Copying the train dataset to the scratch directory...", end=" ")
                shutil.copyfile(self.train_data_dir, destination_train)
                print("DONE")
                print("Extracting the train dataset...", end=" ")
                shutil.unpack_archive(destination_train, os.path.join(self.scratch_path, train_filename.split('.')[-2]))
                print("DONE")

            if not pathlib.Path(os.path.join(self.scratch_path, test_filename.split('.')[-2])).exists():
                destination_test = os.path.join(self.scratch_path, test_filename)
                print("Copying the test dataset to the scratch directory...", end=" ")
                shutil.copyfile(self.test_data_dir, destination_test)
                print("DONE")
                print("Extracting the test dataset...", end=" ")
                shutil.unpack_archive(destination_test, os.path.join(self.scratch_path, test_filename.split('.')[-2]))
                print("DONE")
        
            # change the paths to the scratch directory path
            self.train_data_dir = os.path.join(self.scratch_path, train_filename.split('.')[-2])
            self.test_data_dir = os.path.join(self.scratch_path, test_filename.split('.')[-2])

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

        self.train_dataset = LSV1MDatasetSingleTrial(self.train_data_dir, self.normalize, self.ground_truth_file)
        self.test_dataset = LSV1MDatasetMultiTrial(self.test_data_dir, self.normalize, self.ground_truth_file, self.average_test_trials)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage is None:

            indices = self.train_dataset.indices
            rng = np.random.default_rng(69)
            rng.shuffle(indices)

            subset_idx_val = indices[0 : self.val_size]
            subset_idx_train = indices[self.val_size :]

            self.train_random_sampler = SubsetRandomSampler(subset_idx_train)
            self.train_sequential_sampler = SubsetSequentialSampler(subset_idx_train)
            self.val_sampler = SubsetSequentialSampler(subset_idx_val)

        if stage == "test" or stage is None:
            indices = self.test_dataset.indices
            self.test_sampler = SubsetSequentialSampler(indices)
        
        print("Data loaded successfully!")

    def get_stimulus_visual_angle(self):
        """Returns how much the stimulus spans.

        Returns:
            float: How much the stimulus spans. If the stimulus_crop is defined, then returns
                how much this cropped stimulus spans. The whole span is returned, not to one and other side.
                Returns float as the stimulus is a square.
        """
        return self.stimulus_visual_angle
    
    def get_ground_truth(self, **kwargs):
        """Returns positions in x and y dimensions (in degrees of visual angle) and preferred orientations
           (in radians) of the LSV1M's model's ground truth.
        

        Args:
            ground_truth_positions_file_path (str): Path to the file with positions of neurons
            ground_truth_orientations_file_path (str): Path to the file with preferred orientations of neurons
            in_degrees (Bool, optional): If we want to return the orientations in degrees. If
                False, orientations in radians are returned. Defaults to False.

        Returns:
            tuple: numpy arrays pos_x, pos_y, target_ori (in radians!)
        """
        return self.train_dataset.get_ground_truth(**kwargs)
    
    def visualize_orientation_map(self, save=False, img_path="img/", suffix="_truth", neuron_dot_size=5, in_degrees=False, positions_minus_y=False, positions_minus_x=False, positions_swap_axes=False):
        
        fig, ax = plt.subplots()
        x, y, o = self.get_ground_truth()
        reconstruct_orientation_maps(x, y, o, fig, ax, save, 12, 2.4, 2.4, img_path, suffix, neuron_dot_size)

    def get_input_shape(self):
        x, _ = next(iter(self.train_dataloader()))
        return x[0].shape

    def get_output_shape(self):
        _, y = next(iter(self.train_dataloader()))
        return y[0].shape
    
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
        return self.train_len() + self.val_len()# + self.test_len()

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
        indices = self.test_dataset.indices
        self.test_sampler = SubsetSequentialSampler(indices)
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
        
        model.test_log_name = "val/corr"
        val_score = trainer.test(model, self.val_dataloader(), verbose=False)
        
        model.test_log_name = "test/not_averaged/corr"
        self.test_dataset.average_test_trials = False
        test_score_not_averaged = trainer.test(model, self.test_dataloader(), verbose=False)

        model.test_log_name = "test/averaged/corr"
        self.test_dataset.average_test_trials = True
        test_score_averaged = trainer.test(model, self.test_dataloader(), verbose=False)

        val_score = val_score[0]
        test_score_not_averaged = test_score_not_averaged[0]
        test_score_averaged = test_score_averaged[0]

        print("Validation dataset:")
        print(f"    Correlation: {'{:.4f}'.format(val_score['val/corr'])}")


        # print("Test dataset:")
        # print(f"    Correlation: {'{:.4f}'.format(test_score['test/corr']) }")
        print("Test dataset with NOT averaged responses of repeated trials:")
        print(f"    Correlation: {'{:.4f}'.format(test_score_not_averaged['test/not_averaged/corr'])}")


        print("Test dataset with averaged responses of repeated trials:")
        print(f"    Correlation: {'{:.4f}'.format(test_score_averaged['test/averaged/corr'])}")
        # print(f"    Fraction oracle conservative: {'{:.4f}'.format(test_repeats_averaged_score['test/fraction_oracle_conservative'])} {'({:.2f} percent of the control model)'.format(100 * (test_repeats_averaged_score['test/fraction_oracle_conservative'] / control_measures['test/fraction_oracle_conservative'])) if control_measures else ''}")
        # print(f"    Fraction oracle jackknife: {'{:.4f}'.format(test_repeats_averaged_score['test/fraction_oracle_jackknife'])} {'({:.2f} percent of the control model)'.format(100 * (test_repeats_averaged_score['test/fraction_oracle_jackknife'] / control_measures['test/fraction_oracle_jackknife'])) if control_measures else ''}")

        returned_measures = {
            "val/corr": val_score['val/corr'],
            "test/not_averaged/corr": test_score_not_averaged['test/not_averaged/corr'],
            "test/averaged/corr": test_score_averaged['test/averaged/corr'],
            # "test/repeated_trials/corr": test_repeats_averaged_score['test/repeated_trials/corr'],
            # "test/fraction_oracle_conservative":test_repeats_averaged_score['test/fraction_oracle_conservative'],
            # "test/fraction_oracle_jackknife":test_repeats_averaged_score['test/fraction_oracle_jackknife']
        }

        return returned_measures


    def pickle_read(self, path):
        with open(path, "rb") as f:
            x = pickle.load(f)
        return x

    def get_indices(self, dataset_type="train"):
        """Get indices for possible manual handling of the dataset

        Args:
            dataset_type (str, optional): Type of dataset: "train", "test". Defaults to "train".
        """
        if dataset_type == "train":
            return self.train_dataset.get_indices()
        elif dataset_type == "test":
            return self.test_dataset.get_indices()
        else:
            raise Exception("wrong dataset_type provided in get_indices function")


if __name__ == "__main__":

    path_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/LSV1M/one_trials.pickle"
    path_test = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/LSV1M/ten_trials.pickle"

    path_small_train = "/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/LSV1M/small_train.pickle"

    dm = LSV1MDataModule(path_test, path_test, 10, val_size=500)
    dm.prepare_data()
    dm.setup()
    dm.print_dataset_info()
