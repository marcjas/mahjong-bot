from main import PonNN, DiscardNN
import torch
import torch.nn as nn


class PonModel:
    def __init__(self, model_path):
        self.model = PonNN()
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, X):
        pass


class DiscardModel:
    def __init__(self, model_path):
        self.model = DiscardNN()
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, X):
        X = torch.FloatTensor(X)
        pred = self.model(X)
        pred_probab = nn.Softmax(dim=0)(pred)
        y_pred = pred_probab.argmax(0)
        return tiles[y_pred]

tiles = [
    "1m", "2m", "3m", "4m", 
    "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", 
    "5p", "6p", "7p", "8p", "9p", 
    "1s", "2s", "3s", "4s", 
    "5s", "6s", "7s", "8s", "9s", 
    "ew", "sw", "ww", "nw",
    "wd", "gd", "rd"
]