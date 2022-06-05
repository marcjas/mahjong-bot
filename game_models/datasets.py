import torch
from torch.utils.data import Dataset
import pickle
import sys

chii_data_file_path = "../data/model_data/chii_data.dat"
pon_data_file_path = "../data/model_data/pon_data.dat"
kan_data_file_path = "../data/model_data/kan_data.dat"
riichi_data_file_path = "../data/model_data/riichi_data.dat"
discard_data_file_path = "../data/model_data/discard_data.dat"

class ChiiDataset(Dataset):
    def __init__(self, size=None):
        self.len = Utils.get_metadata()[0] if size == None else size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(chii_data_file_path, shared=False, size=self.len* 4924)).reshape(self.len, 4924)
        print(f"Loaded {self.len} samples...")

    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class PonDataset(Dataset):
    def __init__(self, size=None):
        self.len = Utils.get_metadata()[1] if size == None else size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(pon_data_file_path, shared=False, size=self.len * 4924)).reshape(self.len, 4924)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class KanDataset(Dataset):
    def __init__(self, size=None):
        self.len = Utils.get_metadata()[2] if size == None else size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(kan_data_file_path, shared=False, size=self.len * 4924)).reshape(self.len, 4924)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-1], self.data[idx][-1:]

class RiichiDataset(Dataset):
    def __init__(self, size=None):
        self.len = Utils.get_metadata()[3] if size == None else size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(riichi_data_file_path, shared=False, size=self.len * 4957)).reshape(self.len, 4957)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-34], self.data[idx][-34:]

class DiscardDataset(Dataset):
    def __init__(self, size=None):
        self.len = Utils.get_metadata()[4] if size == None else size
        self.data = torch.FloatTensor(torch.FloatStorage.from_file(discard_data_file_path, shared=False, size=self.len * 4957)).reshape(self.len, 4957)
        print(f"Loaded {self.len} samples...")
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx):
        return self.data[idx][0:-34], self.data[idx][-34:]

class Utils():
    @staticmethod
    def get_metadata():
        try:
            metadata = pickle.load(open("../tenhou_logs_parser/metadata.pickle", "rb"))
        except (OSError, IOError) as error:
            sys.exit("no metadata file")

        return metadata