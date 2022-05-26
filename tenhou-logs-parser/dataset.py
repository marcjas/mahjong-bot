import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
import time

csv_path = "../data/model_data/discard_data.csv"

class DiscardDataset(Dataset):
    def __init__(self, csv_path, chunk_size=None):
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
            "riichi_status"
        ]


        s = time.time()
        self.df = pd.read_csv(csv_path, names=colnames)
        e = time.time()
        print(e - s)

        print(self.df.info())

        # X = self.df.iloc[:,1:3].values

        # test = self.df["private_tiles"]

        
        # print(test)
        # print(self.df)

        # self.X_train = torch.tensor(self.df, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, index):
        return 2
        pass
        # row = torch.as_tensor(self.df.iloc[index], dtype=torch.float32)
        # print("TEST")
        # inputs = tensorData[:, :-1]
        # labels = tensorData[:, 99]
        # return inputs, labels 

def main():
    discard_dataset = DiscardDataset(csv_path)
    # data = DataLoader(dataset=discard_dataset)
            
    # for input, label in enumerate(data):
    #     print(input)
    #     print(label)

if __name__ == '__main__':
    main()