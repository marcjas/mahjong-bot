import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from numpy import save, load

data_file_path = "../data/test/discard_data.dat"

class DiscardDataset(Dataset):
    def __init__(self):
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(data_file_path, shared=False, size=100 * 4957)).reshape(100, 4957)
    
    def __len__(self):
        return 100 

    def __getitem__(self, idx):
        return self.data[idx][0:-34], self.data[idx][-34:]


def main():
    discard_dataset = DiscardDataset()

    print(discard_dataset.__getitem__(5))
    print(len(discard_dataset.__getitem__(5)[0]))
    print(len(discard_dataset.__getitem__(5)[1]))

if __name__ == '__main__':
    main()

