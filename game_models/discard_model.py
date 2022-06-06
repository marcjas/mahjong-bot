import math
from datasets import DiscardDataset
from torch.utils.data import DataLoader 
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import wandb

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

USE_WANDB = False
DATASET_SIZE = 20000 # set to None if you want to load everything
BATCH_SIZE = 8000 
SAVE_INTERVAL = None # set to None if you don't want to torch.save()

# hyperparams
LEARNING_RATE = 1
EPOCHS = 200

if USE_WANDB: wandb.init(project="riichi-mahjong", entity="shuthus")

def main():
    if DATASET_SIZE is not None and BATCH_SIZE > DATASET_SIZE:
        print("BATCH_SIZE can't be smaller than DATASET_SIZE")
        sys.exit(-1)

    dataset = DiscardDataset(DATASET_SIZE)
    dataloader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}...")

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    n_batches = math.ceil(dataset.len / BATCH_SIZE)
    n_size = dataset.len

    for epoch in tqdm(range(EPOCHS)):
        total_epoch_train_loss = 0
        train_correct = 0
        for x, y in dataloader:
            X_train, y_train = x.to(device), y.squeeze().to(device)
            y_pred = net(X_train)

            loss = criterion(y_pred, y_train)
            total_epoch_train_loss += loss
            net.zero_grad() # clear gradients of model parameters
            loss.backward() # update weights
            with torch.no_grad():
                for param in net.parameters():
                    param -= LEARNING_RATE * param.grad

            predictions = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1)
            train_correct += (predictions == y_train).float().sum()

        if USE_WANDB: wandb.log({
            "Epoch": epoch,
            "Train loss": total_epoch_train_loss / n_batches,
            "Train acc": 100 * (train_correct / n_size),
        })

        if SAVE_INTERVAL is not None and (epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1):
            torch.save(net.state_dict(), f"D:/models/discard_model_{epoch}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.ReLU(),
        )

        in_features = 4923
        self.l1 = nn.Sequential(
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Linear(in_features*2, in_features*2),
            nn.ReLU(),
            nn.Linear(in_features*2, 34),
        )

    def forward(self, x):
        return self.l1(x)

if __name__ == "__main__":
    main()

