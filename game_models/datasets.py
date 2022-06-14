import torch
from torch.utils.data import Dataset

CHII_DATA_FILE_PATH = "../data/model_data/chii_data.dat"
PON_DATA_FILE_PATH = "../data/model_data/pon_data.dat"
KAN_DATA_FILE_PATH = "../data/model_data/kan_data.dat"
RIICHI_DATA_FILE_PATH = "../data/model_data/riichi_data.dat"
DISCARD_DATA_FILE_PATH = "../data/model_data/discard_data.dat"

class ChiiDataset(Dataset):
    def __init__(self, size):
        print("Loading data...")
        self.len = size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(CHII_DATA_FILE_PATH, shared=False, size=self.len* 4516)).reshape(self.len, 4516)
        print(f"Loaded {self.len} samples...")

    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class PonDataset(Dataset):
    def __init__(self, size):
        print("Loading data...")
        self.len = size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(PON_DATA_FILE_PATH, shared=False, size=self.len * 4516)).reshape(self.len, 4516)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class KanDataset(Dataset):
    def __init__(self, size):
        print("Loading data...")
        self.len = size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(KAN_DATA_FILE_PATH, shared=False, size=self.len * 4516)).reshape(self.len, 4516)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class RiichiDataset(Dataset):
    def __init__(self, size=None):
        print("Loading data...")
        self.len = Utils.get_metadata()[3] if size == None else size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(RIICHI_DATA_FILE_PATH, shared=False, size=self.len * 4516)).reshape(self.len, 4516)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class DiscardDataset(Dataset):
    def __init__(self, size):
        print("Loading data...")
        self.len = size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(DISCARD_DATA_FILE_PATH, shared=False, size=self.len * 4516)).reshape(self.len, 4516)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:].long()
