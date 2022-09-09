import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from neuralpredictors.data.datasets.statics.filetree import FileTreeDataset
from neuralpredictors.data.transforms import (
    Subsample,
    ToTensor,
    NeuroNormalizer,
    AddBehaviorAsChannels,
    SelectInputChannel,
)
from neuralpredictors.data.samplers import SubsetSequentialSampler

from neuralpredictors.measures.np_functions import (
    oracle_corr_jackknife,
    oracle_corr_conservative,
    fev,
)

from tqdm import tqdm

from typing import Optional

import sys
import zipfile
import pathlib
from distutils.dir_util import copy_tree
import shutil
import wget
import os

# from neuralpredictors.data.samplers import RepeatsBatchSampler
from neuralpredictors.measures.np_functions import corr


from collections import Counter
import numpy as np
from torch.utils.data import Sampler
import torch
import logging

logger = logging.getLogger(__name__)


class RepeatsBatchSampler(Sampler):
    def __init__(self, keys, subset_index=None, **kwargs):
        """A batch sampler where each batch consists of all samples with identical keys value.

        Args:
            keys (Any): list of keys used to group indicies.
            subset_index (list of indices, optional): List of indices to subselect entries from keys.
                            Defaults to None, where all indices from keys are used.
        """
        if subset_index is None:
            subset_index = np.arange(len(keys))
        _, inv = np.unique(
            keys[subset_index], return_inverse=True
        )  # assigns a batch to each trial
        self.repeat_index = np.unique(inv)  # on which index we are at
        self.repeat_sets = inv
        self.subset_index = subset_index
        self.batch_size = 10
        # self.batch_size = self.repeat_sets

    def __iter__(self):
        for u in self.repeat_index:
            self.batch_size = len(list(self.subset_index[self.repeat_sets == u]))
            yield list(self.subset_index[self.repeat_sets == u])

    def __len__(self):
        return len(self.repeat_index)


class LurzDataModule(pl.LightningDataModule):
    """Based on Lurz 2020 code - mouse_loaders.py.
    We work with just one dataset (not multiple), therefore static_loaders were
    ignored, we worked only with ONE static_loader.
    Furthermore, static_shared_loaders were also ignored.

    In this code, there are comments that explain, what exactly was changed.
    The code was updated to work with neuralpredictors version 0.2.0, Lurz's code
    did not work with this version (it worked with 0.0.3 version which is too low)

    This class is an implementation of PyTorch Lightning LightningDataModule,
    so that we can use PyTorch Lightning.
    """

    _URL = "https://gin.g-node.org/cajal/Lurz2020/archive/master.zip"

    def __init__(
        self,
        data_dir,
        batch_size,
        # seed=None,
        areas=None,
        layers=None,
        tier=None,
        neuron_ids=None,
        neuron_n=None,
        exclude_neuron_n=0,
        neuron_base_seed=None,
        image_ids=None,
        image_n=None,
        image_base_seed=None,
        normalize=True,
        include_behavior=False,
        exclude=None,
        select_input_channel=None,
        file_tree=True,
        return_test_sampler=False,
        oracle_condition=None,
        num_workers=0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # seed=None,
        self.areas = areas
        self.layers = layers
        self.tier = tier
        self.neuron_ids = neuron_ids
        self.neuron_n = neuron_n
        self.exclude_neuron_n = exclude_neuron_n
        self.neuron_base_seed = neuron_base_seed
        self.image_ids = image_ids
        self.image_n = image_n
        self.image_base_seed = image_base_seed
        self.normalize = normalize
        self.include_behavior = include_behavior
        self.exclude = exclude
        self.select_input_channel = select_input_channel
        self.file_tree = file_tree
        self.return_test_sampler = return_test_sampler
        self.oracle_condition = oracle_condition
        self.num_workers = num_workers

        # following lines all copied ... just some checks
        assert any(
            [image_ids is None, all([image_n is None, image_base_seed is None])]
        ), "image_ids can not be set at the same time with anhy other image selection criteria"
        assert any(
            [
                self.neuron_ids is None,
                all(
                    [
                        neuron_n is None,
                        neuron_base_seed is None,
                        areas is None,
                        layers is None,
                        exclude_neuron_n == 0,
                    ]
                ),
            ]
        ), "neuron_ids can not be set at the same time with any other neuron selection criteria"
        assert any(
            [exclude_neuron_n == 0, neuron_base_seed is not None]
        ), "neuron_base_seed must be set when exclude_neuron_n is not 0"
        self.data_key = (
            data_dir.split("static")[-1]
            .split(".")[0]
            .replace("preproc", "")
            .replace("_nobehavior", "")
        )

        assert (
            include_behavior and select_input_channel
        ) is False, (
            "Selecting an Input Channel and Adding Behavior can not both be true"
        )

    def prepare_data(self):
        """Downloads the data (if not downloaded yet) and unzips them (if not unzipped yet).

        In this method, as the documentation of Pytorch Lightning states,
        we should not do anything like self.x = y # = assign state
        """
        path = pathlib.Path(self.data_dir)

        if not path.exists():
            print("We have to download and prepare the data.")
            print("This might take a long time (~1 hour).")
            print(
                f"Downloading and extracting the dataset from {self._URL} to folder {self.data_dir}.",
                file=sys.stderr,
            )
            wget.download(self._URL)

            print(
                f"\nUnzipping the dataset to a temporary folder ./tmp_lurz_extracted.",
                file=sys.stderr,
            )
            with zipfile.ZipFile("Lurz2020-master.zip", "r") as zip_ref:
                zip_ref.extractall("tmp_lurz_extracted")

            pathlib.Path(self.data_dir).mkdir(parents=True)

            print(
                f"Copying the temporarily unzipped folder to the specified path",
                file=sys.stderr,
            )
            copy_tree(
                "./tmp_lurz_extracted/lurz2020/static20457-5-9-preproc0", str(path)
            )
            print(
                f"Removing the temporary folder and the downloaded .zip file.",
                file=sys.stderr,
            )
            shutil.rmtree("tmp_lurz_extracted")
            os.remove("Lurz2020-master.zip")

    def setup(self, stage: Optional[str] = None):
        """Sets up the dataset, loads it, shuffles it and sets up the sampler.

        Args:
            stage (Optional[str], optional): Possible values are 'fit, 'test',
            'predict' or None. None means both 'fit' and 'test'. If 'fit', the
            method sets up only the train dataset. If 'test', it sets up only
            the train dataset. If 'predict', it sets up the train dataset.
            Defaults to None.
        """

        # we use only FileTreeDataset (no StaticImageSet) as it was turn on by default
        self.dat = (
            FileTreeDataset(self.data_dir, "images", "responses", "behavior")
            if self.include_behavior
            else FileTreeDataset(self.data_dir, "images", "responses")
        )

        # following lines all copied
        # original comment:
        # The permutation MUST be added first and the conditions below MUST NOT be based on the original order
        # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
        conds = np.ones(len(self.dat.neurons.area), dtype=bool)
        if self.areas is not None:
            conds &= np.isin(self.dat.neurons.area, self.areas)
        if self.layers is not None:
            conds &= np.isin(self.dat.neurons.layer, self.layers)
        idx = np.where(conds)[0]
        if self.neuron_n is not None:
            random_state = np.random.get_state()
            if self.neuron_base_seed is not None:
                np.random.seed(
                    self.neuron_base_seed * self.neuron_n
                )  # avoid nesting by making seed dependent on number of neurons
            assert (
                len(self.dat.neurons.unit_ids) >= self.exclude_neuron_n + self.neuron_n
            ), "After excluding {} neurons, there are not {} neurons left".format(
                self.exclude_neuron_n, self.neuron_n
            )
            self.neuron_ids = np.random.choice(
                self.dat.neurons.unit_ids,
                size=self.exclude_neuron_n + self.neuron_n,
                replace=False,
            )[self.exclude_neuron_n :]
            np.random.set_state(random_state)
        if self.neuron_ids is not None:
            idx = [
                np.where(self.dat.neurons.unit_ids == unit_id)[0][0]
                for unit_id in self.neuron_ids
            ]

        # following lines all copied (but tensors don't go to cuda)
        # transforms .. ToTensor -> no Cuda as it handles the Trainer from Pytorch Lightning
        self.more_transforms = [Subsample(idx), ToTensor(cuda=False)]
        if self.normalize:
            self.more_transforms.insert(
                0, NeuroNormalizer(self.dat, exclude=self.exclude)
            )

        if self.include_behavior:
            self.more_transforms.insert(0, AddBehaviorAsChannels())

        if self.select_input_channel is not None:
            self.more_transforms.insert(
                0, SelectInputChannel(self.select_input_channel)
            )

        # add transforms to the dataset
        self.dat.transforms.extend(self.more_transforms)

        # These two lines changed
        # tier_array = dat.trial_info.tiers if file_tree else dat.tiers
        # image_id_array = dat.trial_info.frame_image_id if file_tree else dat.info.frame_image_id
        # to these lines:
        tier_array = self.dat.trial_info.tiers
        image_id_array = self.dat.trial_info.frame_image_id

        # no return of test_sampler (deleted)

        # the rest of this method was altered a little bit just because of the
        # previous decisions - there are no "tiers", we create train, val test
        # based on stage variable. So only the "decision logic" is a little different
        # (if conditions), but the functionality is basicly the same

        # for creating the separate datasets, we choose ids and create a SubsetSampler
        # that samples the correct images

        # train, test and val datasets are later created based on

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage is None:
            subset_idx_train = None
            subset_idx_val = np.where(tier_array == "validation")[0]

            if self.image_ids is not None:
                subset_idx_train = [
                    np.where(image_id_array == image_id)[0][0]
                    for image_id in self.image_ids
                ]
                assert (
                    sum(tier_array[subset_idx_train] != "train") == 0
                ), "image_ids contain validation or test images"
            elif self.image_n is not None:
                random_state = np.random.get_state()
                if self.image_base_seed is not None:
                    np.random.seed(
                        self.image_base_seed * self.image_n
                    )  # avoid nesting by making seed dependent on number of images
                subset_idx_train = np.random.choice(
                    np.where(tier_array == "train")[0], size=self.image_n, replace=False
                )
                np.random.set_state(random_state)
            else:
                subset_idx_train = np.where(tier_array == "train")[0]

            self.train_random_sampler = SubsetRandomSampler(subset_idx_train)
            self.train_sequential_sampler = SubsetSequentialSampler(subset_idx_train)

            self.val_sampler = SubsetSequentialSampler(subset_idx_val)

        if stage == "test" or stage is None:
            subset_idx_test = np.where(tier_array == "test")[0]
            self.test_sampler = SubsetSequentialSampler(subset_idx_test)

    def get_input_shape(self):
        return self.dat[0].images.shape

    def get_output_shape(self):
        return self.dat[0].responses.shape

    def get_mean_fast(self):
        """Computes the mean response of the validation dataset (much smaller than the train dataset and thus much quicker). If it is available
        in a locally generated file, it loads it from there. Otherwise it
        computes it and then it stores it into the file for the future use.

        Returns:
            torch.Tensor: Mean responses of the neurons.
        """   
        dataloader = DataLoader(
            self.dat, sampler=self.val_sampler, batch_size=self.batch_size
        )
        summed = torch.zeros(self.get_output_shape())
        for d in dataloader:
            summed += torch.sum(d.responses, 0)

        mean = summed / self.train_len()
        return mean

    def get_mean(self):
        """Computes the mean response of the train dataset. If it is available
        in a locally generated file, it loads it from there. Otherwise it
        computes it and then it stores it into the file for the future use.

        Returns:
            torch.Tensor: Mean responses of the neurons.
        """

        mean_path = pathlib.Path(self.data_dir + "/mean.npy")

        if mean_path.exists():
            mean = np.load(mean_path)
            print("Loaded precomputed mean from " + str(mean_path))
            return torch.from_numpy(mean)

        dataloader = DataLoader(
            self.dat, sampler=self.train_sequential_sampler, batch_size=self.batch_size
        )
        summed = torch.zeros(self.get_output_shape())
        for d in dataloader:
            summed += torch.sum(d.responses, 0)

        mean = summed / self.train_len()

        np.save(mean_path, mean)
        print("Created mean array to " + str(mean_path))
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
        return len(self.dat)

    def print_dataset_info(self):
        """Creates a train dataloader, gets first piece of data and prints its shape
        """
        print(" ------------ DATASET INFO ------------ ")
        print(" SHAPES:")
        dataloader = DataLoader(
            self.dat, sampler=self.train_random_sampler, batch_size=self.batch_size
        )
        print(f"    Input shape (images): {self.get_input_shape()}")
        print("    With batch size also: ", end="")
        print(next(iter(dataloader)).images.shape)

        print(f"    Output shape (responses): {self.get_output_shape()}")
        print("    With batch size also: ", end="")
        print(next(iter(dataloader)).responses.shape)

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
            self.dat,
            sampler=self.train_random_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Validation data are in the variable train_data 
        (but the indices are splitted to self.val_sampler)

        Returns:
            DataLoader: The validation DataLoader
        """   
        return DataLoader(
            self.dat,
            sampler=self.val_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Returns:
            DataLoader: The test DataLoader
        """  
        return DataLoader(
            self.dat,
            sampler=self.test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def get_oracle_dataloader(
        self, toy_data=False, oracle_condition=None, verbose=False
        ):
        """Returns the dataset for oracle computations and averaged trials evaluation.

        Args:
            toy_data (bool, optional): Whether to use a small subset of data. Defaults to False.
            oracle_condition (_type_, optional): Conditions to limit the oracle. Defaults to None.
            verbose (bool, optional): Whether to be verbose or not. Defaults to False.

        Raises:
            ValueError: Problems with loading the oracle dataset.

        Returns:
            DataLoader: The test DataLoader with batch_size=10
        """

        if toy_data:
            condition_hashes = self.dat.info.condition_hash
        else:
            dat_info = self.dat.trial_info
            if "image_id" in dir(dat_info):
                condition_hashes = dat_info.image_id
                image_class = dat_info.image_class
            elif "colorframeprojector_image_id" in dir(dat_info):
                condition_hashes = dat_info.colorframeprojector_image_id
                image_class = dat_info.colorframeprojector_image_class
            elif "frame_image_id" in dir(dat_info):
                # gets here in this if else tree
                condition_hashes = dat_info.frame_image_id
                image_class = dat_info.frame_image_class
            else:
                raise ValueError(
                    "'image_id' 'colorframeprojector_image_id', or 'frame_image_id' have to present in the dataset under dat.info "
                    "in order to load get the oracle repeats."
                )

        max_idx = condition_hashes.max() + 1
        classes, class_idx = np.unique(image_class, return_inverse=True)
        identifiers = condition_hashes + class_idx * max_idx

        dat_tiers = self.dat.trial_info.tiers
        sampling_condition = (
            np.where(dat_tiers == "test")[0]
            if oracle_condition is None
            else np.where((dat_tiers == "test") & (class_idx == oracle_condition))[0]
        )
        if (oracle_condition is not None) and verbose:
            print(
                "Created Testloader for image class {}".format(
                    classes[oracle_condition]
                )
            )

        # IDENTIFIERS ARE WHERE THINGS ARE NOT UNIQUE

        sampler = RepeatsBatchSampler(identifiers, sampling_condition)
        return DataLoader(self.dat, batch_sampler=sampler)

    def get_correlations(
        self, model, dataloaders, as_dict=False, per_neuron=True, **kwargs
    ):
        """Computes the correlations. Do not use this method, deprecated.
        """    
        correlations = {}
        for k, v in dataloaders.items():
            target, output = self.model_predictions(
                dataloader=v, model=model, data_key=k
            )
            correlations[k] = corr(target, output, axis=0, eps=1e-12)

            if np.any(np.isnan(correlations[k])):
                print(
                    "{}% NaNs , NaNs will be set to Zero.".format(
                        np.isnan(correlations[k]).mean() * 100
                    )
                )
            correlations[k][np.isnan(correlations[k])] = 0

        if not as_dict:
            correlations = (
                np.hstack([v for v in correlations.values()])
                if per_neuron
                else np.mean(np.hstack([v for v in correlations.values()]))
            )
        return correlations

    def model_predictions(self, model, dataloader, data_key):   
        """
        Computes model predictions for a given dataloader and a model.
        Do not use this method, deprecated.
        Returns:
            tuple: (target, output)
                - target: ground truth, i.e. neuronal firing rates of the neurons
                - output: responses as predicted by the network
        """

        target, output = torch.empty(0), torch.empty(0)
        for images, responses in dataloader:
            # returns in batches.. for example images.shape = [10, 1, 36, 64]
            if len(images.shape) == 5:
                images = images.squeeze(dim=0)
                responses = responses.squeeze(dim=0)
            with torch.no_grad():
                # the outputs and targets get concatenated into a veeeery long tensor
                output = torch.cat((output, (model(images).detach().cpu())), dim=0)
                target = torch.cat((target, responses.detach().cpu()), dim=0)

        # on oracle dataloader returns both of shape (999, 5335), because 999 is test_len and 5335 is number_of_neurons
        # or for dm.val_dataloader() returns (522, 5335), where 522 is dm.val_len()
        return target.numpy(), output.numpy()

    def return_test_sampler(self, oracle_condition=None, get_key=False):
        """

        Args:
            oracle_condition (_type_, optional): To limit the oracle. Defaults to None.
            get_key (bool, optional): _description_. Defaults to False.

        Returns:
            tuple or dict: The test sampler
        """        
        print("Returning only test sampler with repeats...")
        dataloader = self.get_oracle_dataloader(oracle_condition=oracle_condition)
        return (
            (self.data_key, {"test": dataloader}) if get_key else {"test": dataloader}
        )

    def predict_dataloader(self):
        """
        Returns:
            DataLoader: The test DataLoader
        """   
        return DataLoader(
            self.dat,
            sampler=self.test_sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        

if __name__ == "__main__":

    data_dir = "data/lurz2020/static20457-5-9-preproc0"
    dataset_config = {"data_dir": data_dir, "batch_size": 10, "normalize": True}
    #   "exclude": None}

    dm = LurzDataModule(**dataset_config)
    dm.prepare_data()
    dm.setup()
    dm.print_dataset_info()
