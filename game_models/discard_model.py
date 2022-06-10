import math
from datasets import DiscardDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import wandb
import time
import os

# config
USE_WANDB = False
SAVE_INTERVAL = None # set to None if you don't want to torch.save()
SAVE_PATH = "D:/models"
RNG_SEED = None
TRAIN_DATASET_SIZE = 1000
TEST_DATASET_SIZE = 100
BATCH_SIZE = 1000

# hyperparams
LEARNING_RATE = 1
EPOCHS = 200

if USE_WANDB: wandb.init(project="riichi-mahjong", entity="shuthus", tags=["discard"])

def main():
    if TRAIN_DATASET_SIZE and BATCH_SIZE > TRAIN_DATASET_SIZE:
        print("BATCH_SIZE can't be bigger than TRAIN_DATASET_SIZE")
        sys.exit(-1)

    if RNG_SEED: torch.manual_seed(RNG_SEED)

    dataset = DiscardDataset(TRAIN_DATASET_SIZE + TEST_DATASET_SIZE)
    train_dataset, test_dataset = random_split(dataset, (TRAIN_DATASET_SIZE, TEST_DATASET_SIZE))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train size: {len(train_dataset)}, test size: {len(test_dataset)}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    train_n_batches = math.ceil(len(train_dataset) / BATCH_SIZE)
    train_n_size = len(train_dataset)
    test_n_size = len(test_dataset)

    timestamp = int(time.time())

    for epoch in tqdm(range(1, EPOCHS + 1)):
        total_epoch_train_loss = 0
        train_correct = 0
        for x, y in train_dataloader:
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

        if epoch % 10 == 0 or epoch == EPOCHS:
            test_correct = 0
            for x, y in test_dataloader:
                X_test, y_test = x.to(device), y.squeeze().to(device)
                y_pred = net(X_test)

                predictions = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1)
                test_correct += (predictions == y_test).float().sum()

            if USE_WANDB: wandb.log({
                "Epoch": epoch,
                "Train loss": total_epoch_train_loss / train_n_batches,
                "Train acc": 100 * (train_correct / train_n_size),
                "Test acc": 100 * (test_correct / test_n_size),
            })

        if SAVE_INTERVAL and (epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS):
            fn = f"{SAVE_PATH}/discard_{timestamp}_{epoch}.pt"
            torch.save(net.state_dict(), fn)
            print(f"Saved model {fn} with size {os.path.getsize(fn)}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        in_features = 4923
        self.l1 = nn.Sequential(
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Linear(in_features*2, 34),
        )

    def forward(self, x):
        return self.l1(x)

class DiscardModel:
    def __init__(self, model_path):
        self.model = Net().load_state_dict(torch.load(model_path))

    def predict(self, X):
        X = torch.FloatTensor(X)
        pred = self.model(X)
        pred_probab = nn.Softmax(dim=1)(pred)
        y_pred = pred_probab.argmax(1)
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

if __name__ == "__main__":
    main()

