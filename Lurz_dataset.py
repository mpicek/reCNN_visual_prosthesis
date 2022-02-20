import pytorch_lightning as pl
from torch.utils.data import DataLoader

from collections import OrderedDict
from itertools import zip_longest
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from neuralpredictors.data.datasets.statics.filetree import FileTreeDataset
from neuralpredictors.data.transforms import (
    Subsample,
    ToTensor,
    NeuroNormalizer,
    AddBehaviorAsChannels,
    SelectInputChannel,
)
from neuralpredictors.data.samplers import SubsetSequentialSampler


# TODO: normalizaci a pripadne dalsi transformace dat do transforms
class LurzDataModule(pl.LightningDataModule):
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
        oracle_condition=None
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
        
        assert any(
        [image_ids is None, all([image_n is None, image_base_seed is None])]
        ), "image_ids can not be set at the same time with anhy other image selection criteria"
        assert any(
            [
                neuron_ids is None,
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
        pass


    def setup(self, stage: Optional[str] = None):
        # stage is "fit" or "test"
        # when stage=None -> both "fit" and "test"

        self.dat = (
            FileTreeDataset(self.data_dir, "images", "responses", "behavior")
            if self.include_behavior
            else FileTreeDataset(self.data_dir, "images", "responses")
        )

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
            neuron_ids = np.random.choice(self.dat.neurons.unit_ids, size=self.exclude_neuron_n + self.neuron_n, replace=False)[
                self.exclude_neuron_n:
            ]
            np.random.set_state(random_state)
        if neuron_ids is not None:
            idx = [np.where(self.dat.neurons.unit_ids == unit_id)[0][0] for unit_id in neuron_ids]

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

        tier_array = self.dat.trial_info.tiers
        image_id_array = self.dat.trial_info.frame_image_id



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
            

            self.train_sampler = SubsetRandomSampler(subset_idx_train)
            self.val_sampler = SubsetSequentialSampler(subset_idx_val)
        
        if stage == "test" or stage is None:
            subset_idx_test = np.where(tier_array == "test")[0]
            self.test_sampler = SubsetSequentialSampler(subset_idx_test)


    def train_dataloader(self):
        return DataLoader(self.dat, sampler=self.train_sampler, batch_size=self.batch_size) #TODO: shuffle=True??? https://github.com/PyTorchLightning/pytorch-lightning/discussions/7332

    def val_dataloader(self):
        return DataLoader(self.dat, sampler=self.val_sampler, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dat, sampler=self.test_sampler, batch_size=self.batch_size)

    def predict_dataloader(self):
        # TODO: return some separate subset for prediction and not only test
        return DataLoader(self.dat, sampler=self.test_sampler, batch_size=self.batch_size)