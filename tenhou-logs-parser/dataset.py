import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

csv_path = "../data/model_data/discard_data.csv"

class DiscardDataset(Dataset):
    def __init__(self, csv_path):
        colnames = [
            "private_tiles", 
            "private_discarded_tiles", 
            "others_discarded_tiles", 
            "private_open_tiles", 
            "others_open_tiles", 
            "dora_indicators",
            "round_name",
            "player_scores",
            "self_wind",
            "aka_doras_in_hand",
            "riichi_status",
            "discarded_tile"
        ]

        self.df = pd.read_csv(csv_path, names=colnames, nrows=100)

        self.preprocess_features()
        self.preprocess_labels() 

        # self.X = torch.tensor(self.df[:-1], dtype=torch.float32)
        self.y = torch.tensor(self.df["discarded_tile"], dtype=torch.uint8)

        # self.X = []
        # self.y = []

    def preprocess_labels(self):
        self.df["discarded_tile"] = self.df["discarded_tile"].apply(self.tile_to_channel)

    def preprocess_features(self):

        pass

    def tile_to_channel(self, tile):
        channel = [0] * 34
        channel[TILES.index(tile[0:2])] = 1
        return channel

    
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

TILES = """
    1m 2m 3m 4m 5m 6m 7m 8m 9m
    1p 2p 3p 4p 5p 6p 7p 8p 9p
    1s 2s 3s 4s 5s 6s 7s 8s 9s
    ew sw ww nw
    wd gd rd
""".split()

def main():
    discard_dataset = DiscardDataset(csv_path)
    # data = DataLoader(dataset=discard_dataset)
            
    # for x, y in enumerate(data):
    #     print(x)
    #     print(y)

if __name__ == '__main__':
    main()