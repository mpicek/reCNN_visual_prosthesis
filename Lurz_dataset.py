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


# TODO: normalizaci a pripadne dalsi transformace dat do transforms
class LurzDataModule(pl.LightningDataModule):
    """
    Based on Lurz 2020 code - mouse_loaders.py.
    We work with just one dataset (not multiple), therefore static_loaders were
    ignored, we worked only with ONE static_loader.
    Furthermore, static_shared_loaders were also ignored.

    In this code, there are comments that explain, what exactly was changed.
    The code was updated to work with neuralpredictors version 0.2.0, Lurz's code
    did not work with this version (it worked with 0.0.3 version)

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
        exclude="images",
        select_input_channel=None,
        file_tree=True,
        return_test_sampler=False,
        oracle_condition=None,
        num_workers=0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # seed=None,
        self.areas=areas
        self.layers=layers
        self.tier=tier
        self.neuron_ids=neuron_ids
        self.neuron_n=neuron_n
        self.exclude_neuron_n=exclude_neuron_n
        self.neuron_base_seed=neuron_base_seed
        self.image_ids=image_ids
        self.image_n=image_n
        self.image_base_seed=image_base_seed
        self.normalize=normalize
        self.include_behavior=include_behavior
        self.exclude=exclude
        self.select_input_channel=select_input_channel
        self.file_tree=file_tree
        self.return_test_sampler=return_test_sampler
        self.oracle_condition=oracle_condition
        self.num_workers=num_workers
        
        # following lines all copied ... just some checks
        assert any(
        [image_ids is None, all([image_n is None, image_base_seed is None])]
        ), "image_ids can not be set at the same time with anhy other image selection criteria"
        assert any(
            [
                self.neuron_ids is None,
                all([neuron_n is None, neuron_base_seed is None, areas is None, layers is None, exclude_neuron_n == 0]),
            ]
        ), "neuron_ids can not be set at the same time with any other neuron selection criteria"
        assert any(
            [exclude_neuron_n == 0, neuron_base_seed is not None]
        ), "neuron_base_seed must be set when exclude_neuron_n is not 0"
        data_key = data_dir.split("static")[-1].split(".")[0].replace("preproc", "").replace("_nobehavior", "")

        assert (
            include_behavior and select_input_channel
        ) is False, "Selecting an Input Channel and Adding Behavior can not both be true"


    def prepare_data(self):
        # we should not do anything like self.x = y # = assign state
        # just download the data
        path = pathlib.Path(self.data_dir)
        
        if not path.exists():
            print("We have to download and prepare the data.");
            print("This might take a long time (~1 hour).")
            print(f"Downloading and extracting the dataset from {self._URL} to folder {self.data_dir}.", file=sys.stderr)
            wget.download(self._URL)

            print(f"\nUnzipping the dataset to a temporary folder ./tmp_lurz_extracted.", file=sys.stderr)
            with zipfile.ZipFile("Lurz2020-master.zip", "r") as zip_ref:
                zip_ref.extractall("tmp_lurz_extracted")

            pathlib.Path(self.data_dir).mkdir(parents=True)

            print(f"Copying the temporarily unzipped folder to the specified path", file=sys.stderr)
            copy_tree("./tmp_lurz_extracted/lurz2020/static20457-5-9-preproc0", str(path))
            print(f"Removing the temporary folder and the downloaded .zip file.", file=sys.stderr)
            shutil.rmtree("tmp_lurz_extracted")
            os.remove("Lurz2020-master.zip")


    def setup(self, stage: Optional[str] = None):
        # stage is "fit" or "test" or "predict"
        # when stage=None -> both "fit" and "test"

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
                np.random.seed(self.neuron_base_seed * self.neuron_n)  # avoid nesting by making seed dependent on number of neurons
            assert (
                len(self.dat.neurons.unit_ids) >= self.exclude_neuron_n + self.neuron_n
            ), "After excluding {} neurons, there are not {} neurons left".format(self.exclude_neuron_n, self.neuron_n)
            self.neuron_ids = np.random.choice(self.dat.neurons.unit_ids, size=self.exclude_neuron_n + self.neuron_n, replace=False)[
                self.exclude_neuron_n:
            ]
            np.random.set_state(random_state)
        if self.neuron_ids is not None:
            idx = [np.where(self.dat.neurons.unit_ids == unit_id)[0][0] for unit_id in self.neuron_ids]

        # following lines all copied (but tensors don't go to cuda)
        # transforms .. ToTensor -> no Cuda as it handles the Trainer from Pytorch Lightning
        self.more_transforms = [Subsample(idx), ToTensor(cuda=False)]
        if self.normalize:
            self.more_transforms.insert(0, NeuroNormalizer(self.dat, exclude=self.exclude))

        if self.include_behavior:
            self.more_transforms.insert(0, AddBehaviorAsChannels())

        if self.select_input_channel is not None:
            self.more_transforms.insert(0, SelectInputChannel(self.select_input_channel))
        
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
        if stage == "fit" or stage == "predict" or stage is None: # TODO: predict make separate 
            subset_idx_train = None
            subset_idx_val = np.where(tier_array == "validation")[0]
            

            if self.image_ids is not None:
                subset_idx_train = [np.where(image_id_array == image_id)[0][0] for image_id in self.image_ids]
                assert sum(tier_array[subset_idx_train] != "train") == 0, "image_ids contain validation or test images"
            elif self.image_n is not None:
                random_state = np.random.get_state()
                if self.image_base_seed is not None:
                    np.random.seed(self.image_base_seed * self.image_n)  # avoid nesting by making seed dependent on number of images
                subset_idx_train = np.random.choice(np.where(tier_array == "train")[0], size=self.image_n, replace=False)
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
        """ Computes the mean response of the validation dataset (much smaller than the train set) """
        dataloader = DataLoader(self.dat, sampler=self.val_sampler, batch_size=self.batch_size)
        summed = torch.zeros(self.get_output_shape())
        for d in dataloader:
            summed += torch.sum(d.responses, 0)

        mean = summed / self.train_len()
        return mean
    
    def get_mean(self):
        """ Computes the mean response of the train dataset """

        dataloader = DataLoader(self.dat, sampler=self.train_sequential_sampler, batch_size=self.batch_size)
        summed = torch.zeros(self.get_output_shape())
        for d in dataloader:
            summed += torch.sum(d.responses, 0)

        mean = summed / self.train_len()
        return mean
    
    def train_len(self):
        return len(self.train_random_sampler)
    
    def val_len(self):
        return len(self.val_sampler)
    
    def test_len(self):
        return len(self.test_sampler)
    
    def __len__(self):
        """The length of ALL the data we have (train + val + test)"""
        return len(self.dat)

    def print_dataset_info(self):
        """
        Creates a train dataloader, gets first piece of data and prints its shape
        """
        print(" ------------ DATASET INFO ------------ ")
        print(" SHAPES:")
        dataloader = DataLoader(self.dat, sampler=self.train_random_sampler, batch_size=self.batch_size) #TODO: shuffle=True??? https://github.com/PyTorchLightning/pytorch-lightning/discussions/7332
        print(f"    Input shape (images): {self.get_input_shape()}")
        print("    With batch size also: ", end='')
        print(next(iter(dataloader)).images.shape)

        print(f"    Output shape (responses): {self.get_output_shape()}")
        print("    With batch size also: ", end='')
        print(next(iter(dataloader)).responses.shape)

        print(" LENGTH:")
        print(f"    Length of the dataset is {len(self)}")
        print(f"    Length of the train set is {self.train_len()}")
        print(f"    Length of the val set is {self.val_len()}")
        print(f"    Length of the test set is {self.test_len()}")
        
        print(" -------------------------------------- ")
        

    def train_dataloader(self):
        return DataLoader(self.dat, sampler=self.train_random_sampler, batch_size=self.batch_size, num_workers=self.num_workers) #TODO: shuffle=True??? https://github.com/PyTorchLightning/pytorch-lightning/discussions/7332

    def val_dataloader(self):
        return DataLoader(self.dat, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dat, sampler=self.test_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        # TODO: return some separate subset for prediction and not only test
        return DataLoader(self.dat, sampler=self.test_sampler, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def model_performances(self, model=None):
        test = self.test_dataloader()
        #TODO
        for d in tqdm(test):
            pprint(d)

