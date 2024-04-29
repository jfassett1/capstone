import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import re
from pathlib import Path
from collections import deque
import pathlib
# def get_num_rows(npy_file):
#     with np.load(npy_file, mmap_mode='r') as data:
#         num_rows = data.shape[0]
#     return num_rows


def get_num_rows(npy_file):
    data = np.load(npy_file, mmap_mode='r')  #Load the file with memory-mapping in read mode
    num_rows = data.shape[0]  #Gets number of rows
    del data  #Deleting data to free up memory
    return num_rows


class CLSdata(Dataset):
    def __init__(self, csv_file, npy_dir,vocal=False):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.labels = torch.tensor(self.df['troll'].values, dtype=torch.long)
        #Deleting unneeded columns
        cols = set(self.df.columns)
        metadata = set(['followers','retweet','hash','link','capitals','numerics','special_chars','exclamations','avg_word_length','sentences'])
        unneeded = cols - metadata
        self.df.drop(columns=unneeded,inplace=True)

        # self.df.drop(columns=["Unnamed: 0", "content"], inplace=True)
        # Efficiently load and process numpy files
        self.npy_files = list(Path(npy_dir).glob("*.npy"))
        self.npy_sizes = [get_num_rows(file) for file in self.npy_files]
        # npylist = [np.load(os.path.join(npy_dir, file)) for file in self.npy_files]
        # self.data = torch.tensor(np.concatenate(npylist, axis=0), dtype=torch.float32).squeeze(1)
        print(self.npy_sizes)
        self.curridx = -1
        self.vocal = vocal
        # print(f'Number of npy files: {len(self.npy_files)}', flush=True)
        # print(len(self.npy_sizes))
    def __len__(self):
        return len(self.npy_files) * self.npy_sizes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.vocal:
            print(f"Requesting item {idx}")
        
        # Calculate the cumulative size to find the correct .npy file and local index
        cumulative_sizes = np.cumsum([0] + self.npy_sizes)
        filereq = np.searchsorted(cumulative_sizes, idx+1, side='right') - 1

        if filereq != self.curridx:
            self.curridx = filereq
            self.curr = np.load(self.npy_files[filereq])
        
        localidx = idx - cumulative_sizes[filereq]
        print(filereq)
        embed = torch.tensor(self.curr[localidx], dtype=torch.float32)
        label = self.labels[idx]
        return embed, label


def numerical_sort(file):
    """ Extracts numbers from filename and returns them as an integer for sorting purposes. """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(file.name)
    parts[1::2] = map(int, parts[1::2])  # Convert the extracted numbers to integers
    return parts

class CLSdata2(Dataset):
    def __init__(self, csv_file, npy_dir, vocal=False, cache_size=5):
        super().__init__()

        self.npy_files = sorted(Path(npy_dir).glob("*.npy"),key=numerical_sort)
        self.npy_sizes = [np.load(file, mmap_mode='r').shape[0] for file in self.npy_files]
        self.length = sum(self.npy_sizes) -1
        self.cumulative_sizes = np.cumsum([0] + self.npy_sizes)
        self.vocal = vocal
        self.cache_size = cache_size
        self.file_cache = {}
        self.cache_keys = deque()

        self.df = pd.read_csv(csv_file).iloc[:self.length]
        self.labels = torch.tensor(self.df['troll'].values, dtype=torch.long)
        self.df.drop(columns=["Unnamed: 0", "content"], inplace=True)

    def __len__(self):
        return self.length

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
            # if file_idx == len(self.npy_files):
            #     print("2 high")
            #     file_idx = len(self.npy_files) - 1
                
            self.file_cache[file_idx] = np.load(self.npy_files[file_idx], mmap_mode='r')
            self.cache_keys.append(file_idx)
        local_idx = idx - self.cumulative_sizes[file_idx]
        embed = torch.from_numpy(self.file_cache[file_idx][local_idx]).float()
        label = self.labels[idx]
        
        return embed, label
    
if __name__ == "__main__":  
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    # Usage
    # csv_file = pathlib.Path()
    npy_dir = 'src/embedded/'

    my_dataset = CLSdata2(csv_file=csv_file, npy_dir=npy_dir)
