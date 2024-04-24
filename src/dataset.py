
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import logging
from pathlib import Path
from collections import deque
from utils import CLSdata2
from tqdm import tqdm

class TweetDataset(Dataset):
    def __init__(self, npy_dir, vocal=False, cache_size=5):
        super().__init__()
        self.labels = np.load(npy_dir/"labels.npy")
        self.npy_files = sorted(Path(npy_dir).glob("*.npy"))
        self.npy_sizes = [np.load(file, mmap_mode='r').shape[0] for file in self.npy_files]
        self.cumulative_sizes = np.cumsum([0] + self.npy_sizes)
        
        self.vocal = vocal
        self.cache_size = cache_size
        self.file_cache = {}
        self.cache_keys = deque()

    def __len__(self):
        return len(self.labels) - 3

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()        
        file_idx = np.searchsorted(self.cumulative_sizes, idx + 1, side='right') - 1
        if file_idx not in self.file_cache:
            if self.vocal:
                print("Loading new file...")
            if len(self.cache_keys) >= self.cache_size:
                old_key = self.cache_keys.popleft()
                del self.file_cache[old_key]
            self.file_cache[file_idx] = np.load(self.npy_files[file_idx], mmap_mode='r')
            self.cache_keys.append(file_idx)
        
        local_idx = idx - self.cumulative_sizes[file_idx]
        array = self.file_cache[file_idx][local_idx]
        array_copy = np.copy(array)
        embed = torch.from_numpy(array_copy).float()
        label = self.labels[idx]
        
        return embed, label
    

class TweetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size=64, vocal=False, cache_size=5):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.vocal = vocal
        self.cache_size = cache_size
        self.train_dataset = TweetDataset(dataset_dir/"train")
        self.val_dataset = TweetDataset(dataset_dir/"val")
        self.test_dataset = TweetDataset(dataset_dir/"test")

    def setup(self, stage=None):
        # Adjust these splits as necessary
        print("nuts")
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Implement this if you have test data
        pass
if __name__ == "__main__":
    # csv_file = 'transformed/data.csv'
    # npy_dir = 'embedded/'
    which = "train"
    data_dir = Path(__file__).parent.parent/"data"
    npy_dir = data_dir/"dataset"/which


    dm = TweetDataModule(data_dir/"dataset")

    for i in tqdm(dm.val_dataloader(),colour="blue"):
        pass
    # print(type(dataset[5][0].numpy()))
    # exit()



    # data_dir = Path("C:/Users/Jayden/Documents/school/capstone/data/")



    # datapth = data_dir/"dataset"/which/"labels.npy"


