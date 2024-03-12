import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path

# def get_num_rows(npy_file):
#     with np.load(npy_file, mmap_mode='r') as data:
#         num_rows = data.shape[0]
#     return num_rows

class SimpleCLSdata(Dataset):
    def __init__(self, npy_dir):
        super().__init__()
        npy_file = list(Path(npy_dir).glob("*.npy"))[0]  # Just take the first npy file for testing
        data = np.load(npy_file, mmap_mode='r')  # Attempt to load it
        print(data.shape)  # Print shape to confirm it's loaded correctly

    def __len__(self):
        return 1  # Return a dummy length

    def __getitem__(self, idx):
        return torch.tensor([1.0])  # Return a dummy item




def get_num_rows(npy_file):
    data = np.load(npy_file, mmap_mode='r')  # Load the file with memory-mapping in read mode
    num_rows = data.shape[0]  # Access the shape to get the number of rows
    del data  # This line attempts to ensure that any resources are freed, though actual cleanup is dependent on Python's garbage collection
    return num_rows

class CLSdata(Dataset):
    def __init__(self, csv_file, npy_dir,vocal=False):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.labels = torch.tensor(self.df['troll'].values, dtype=torch.long) # Convert labels to tensor
        self.df.drop(columns=["Unnamed: 0", "content"], inplace=True)
        # Efficiently load and process numpy files
        self.npy_files = list(Path(npy_dir).glob("*.npy"))
        self.npy_sizes = [get_num_rows(file) for file in self.npy_files]
        # npylist = [np.load(os.path.join(npy_dir, file)) for file in self.npy_files]
        # self.data = torch.tensor(np.concatenate(npylist, axis=0), dtype=torch.float32).squeeze(1) # Load, concatenate, and convert to tensor
        self.curr = None
        self.curridx = -1
        self.vocal = vocal
        # print(f'Number of npy files: {len(self.npy_files)}', flush=True)
        # print(len(self.npy_sizes))
    def __len__(self):
        return len(self.df)

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
            print(filereq)
            self.curr = np.load(self.npy_files[filereq])
        
        localidx = idx - cumulative_sizes[filereq]
        
        embed = torch.tensor(self.curr[localidx], dtype=torch.float32)  # Convert to tensor
        label = self.labels[idx]  # Use preloaded tensor labels
        return embed, label

# Usage
# csv_file = 'src/transformed/data.csv'
# npy_dir = 'src/embedded/'

# my_dataset = MyDataset(csv_file=csv_file, npy_dir=npy_dir)
# dataloader = DataLoader(my_dataset, batch_size=4, shuffle=True)


# def dataloader():
#     # Initialize an empty array for more efficient concatenation
#     cols_list = []
    
#     for file in os.listdir("src/embedded/"):
#         if file.endswith(".npy"):
#             filepath = os.path.join("src/embedded/", file)
#             # Load each file and append to list without concatenating immediately
#             cols_list.append(np.load(filepath))
    
#     # Concatenate all arrays at once, more memory efficient than appending in a loop
#     cols = np.concatenate(cols_list, axis=0)
#     cols = np.squeeze(cols, axis=1)
    
#     # Efficient loading of csv
#     df = pd.read_csv("src/transformed/data.csv", usecols=lambda column: column not in ["Unnamed: 0"])
    
#     # Create DataFrame from Numpy Array
#     bertdf = pd.DataFrame(cols)
    
#     # Concatenate DataFrames
#     ml_df = pd.concat([df, bertdf], axis=1)