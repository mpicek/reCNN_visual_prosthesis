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


class AntolikDataset(Dataset):
    def __init__(self, path, normalize=True):
        self.normalize = normalize

        self.data = self.pickle_read(path)
        
        self.transform_list = transforms.Compose([transforms.Normalize((45.2315, ), (26.6845, ))])
        
    def __getitem__(self, index):
        x = self.data[index]["stimulus"]
        x = np.expand_dims(x, axis=0)
        y = np.concatenate([self.data[index]["V1_Exc_L2/3"], self.data[index]["V1_Inh_L2/3"]])
        

        data = torch.from_numpy(x)
        target = torch.from_numpy(y)

        if self.normalize:
            data = self.transform_list(data)

        return (data.float(), target.float())
    
    def __len__(self):
        return len(self.data)
    
    def pickle_read(self, path):
        with open(path, "rb") as f:
            x = pickle.load(f)
        return x


class AntolikDataModule(pl.LightningDataModule):
    """
    Has to be downloaded from the wintermute server.
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
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size
        # seed=None,
        self.normalize=normalize
        self.num_workers=num_workers
        self.val_size = val_size


    def prepare_data(self):
        """
        We do not have public access to the data. This function will be implemented
        when the dataset is available on some website.
        """
        # we should not do anything like self.x = y # = assign state
        # just download the data
        train_path = pathlib.Path(self.train_data_dir)
        test_path = pathlib.Path(self.test_data_dir)
        
        if not train_path.exists():
            raise Exception("The .pickle file with Antolik train dataset does not exist.")

        if not test_path.exists():
            raise Exception("The .pickle file with Antolik test dataset does not exist.")


    def setup(self, stage: Optional[str] = None):
        # stage is "fit" or "test" or "predict"
        # when stage=None -> both "fit" and "test"
        

        self.train_dataset = AntolikDataset(self.train_data_dir, normalize=self.normalize)
        self.train_data = self.pickle_read(self.train_data_dir)
        self.test_dataset = AntolikDataset(self.test_data_dir, normalize=self.normalize)
        self.test_data = self.pickle_read(self.test_data_dir)

        print("Data loaded successfully!")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage == "predict" or stage is None: # TODO: predict make separate

            indices = np.arange(0, len(self.train_data))

            rng = np.random.default_rng(69)
            rng.shuffle(indices)
            indices_keys = [list(self.train_data.keys())[i] for i in indices]

            subset_idx_val = indices_keys[0:self.val_size]
            subset_idx_train = indices_keys[self.val_size:]

            self.subset_idx_val = subset_idx_val
            
            self.train_random_sampler = SubsetRandomSampler(subset_idx_train)
            self.train_sequential_sampler = SubsetSequentialSampler(subset_idx_train)
            self.val_sampler = SubsetSequentialSampler(subset_idx_val)

        
        if stage == "test" or stage is None:
            indices = np.arange(0, len(self.test_data))
            subset_idx_test = [list(self.test_data.keys())[i] for i in indices]
            self.test_sampler = SubsetSequentialSampler(subset_idx_test)

    def get_input_shape(self):
        x, _ = next(iter(self.train_dataloader()))
        return x[0].shape

    def get_output_shape(self):
        _, y = next(iter(self.train_dataloader()))
        return y[0].shape

    def get_mean(self):
        """ Computes the mean response of the train dataset """
        
        mean_path = pathlib.Path(self.train_data_dir.rsplit('.pickle', 1)[0] + "_mean.npy")

        if mean_path.exists():
            mean = np.load(mean_path)
            print("Loaded precomputed mean from " + str(mean_path))
            return torch.from_numpy(mean)

        dataloader = DataLoader(self.train_dataset, sampler=self.train_sequential_sampler, batch_size=self.batch_size)
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
        """The length of ALL the data we have (train + val + test)"""
        return self.train_len() + self.val_len() + self.test_len()

    def print_dataset_info(self):
        """
        Creates a train dataloader, gets first piece of data and prints its shape
        """
        print(" ------------ DATASET INFO ------------ ")
        print(" SHAPES:")
        dataloader = DataLoader(self.train_dataset, sampler=self.train_random_sampler, batch_size=self.batch_size)
        print(f"    Input shape (images): {self.get_input_shape()}")
        print("    With batch size also: ", end='')
        print(next(iter(dataloader))[0].shape)

        print(f"    Output shape (responses): {self.get_output_shape()}")
        print("    With batch size also: ", end='')
        print(next(iter(dataloader))[1].shape)

        print(" LENGTH:")
        print(f"    Length of the dataset is {len(self)}")
        print(f"    Length of the train set is {self.train_len()}")
        print(f"    Length of the val set is {self.val_len()}")
        print(f"    Length of the test set is {self.test_len()}")
        
        print(" -------------------------------------- ")
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, sampler=self.train_random_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # validation data are in the variable train_data (but the indices are splitted to self.val_sampler)
        return DataLoader(self.train_dataset, sampler=self.val_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_oracle_dataloader(self):
        # the only difference from test_dataloader is that we hardcode batch_size=10
        return DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=10, num_workers=self.num_workers)

    def model_performances(self, model=None, trainer=None):
        model.test_average_batch = False
        model.compute_oracle_fraction = False
        val_score = trainer.test(model, self.val_dataloader())
        test_score = trainer.test(model, self.test_dataloader())

        model.test_average_batch = True
        model.compute_oracle_fraction = True
        test_repeats_averaged_score = trainer.test(model, self.get_oracle_dataloader())

        from pprint import pprint
        print("val_score")
        pprint(val_score)
        print("test_score")
        pprint(test_score)
        print("test_repeats_averaged_score")
        pprint(test_repeats_averaged_score)
    
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

