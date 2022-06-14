import argparse
import math
from datasets import ChiiDataset, PonDataset, KanDataset, RiichiDataset, DiscardDataset
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import time
import os

# config
USE_WANDB = True
SAVE_INTERVAL = 200 
SAVE_PATH = "../models"
RNG_SEED = 1234 
TRAIN_DATASET_SIZE = 20000
TEST_DATASET_SIZE = 1000 
BATCH_SIZE = 7000 

# hyperparams
LEARNING_RATE = 1
EPOCHS = 200

def main(model_type):
    if RNG_SEED: torch.manual_seed(RNG_SEED)

    train_dataloader, test_dataloader = get_dataloaders(model_type)

    if USE_WANDB: 
        config = {
            "LR": LEARNING_RATE,
            "TRAIN_SIZE": TRAIN_DATASET_SIZE,
            "TEST_SIZE": TEST_DATASET_SIZE
        }
        wandb.init(project="riichi-mahjong", entity="shuthus", tags=[model_type], config=config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    match model_type:
        case ("chii" | "pon" | "kan"): create_call_model(model_type, device, train_dataloader, test_dataloader)
        case "riichi": create_riichi_model(device, train_dataloader, train_dataloader)
        case "discard": create_discard_model(device, train_dataloader, test_dataloader)

def create_call_model(model_type, device, train_dataloader, test_dataloader, start_time=int(time.time())):
    net = CallNN().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    train_n_batches = math.ceil(TRAIN_DATASET_SIZE / BATCH_SIZE)

    for epoch in tqdm(range(1, EPOCHS + 1)):
        total_epoch_train_loss = 0
        train_correct = 0
        for x, y in train_dataloader:
            X_train, y_train = x.to(device), y.to(device)
            y_pred = net(X_train)

            loss = criterion(y_pred, y_train)
            total_epoch_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == EPOCHS:
                predictions = (y_pred > 0.5).float()*1
                train_correct += (predictions == y_train).float().sum()

        if epoch % 10 == 0 or epoch == EPOCHS:
            test_correct = 0
            for x, y in test_dataloader:
                X_test, y_test = x.to(device), y.to(device)
                y_pred = net(X_test)

                predictions = (y_pred > 0.5).float()*1
                test_correct += (predictions == y_test).float().sum()

            if USE_WANDB: wandb.log({
                "Epoch": epoch,
                "Train loss": total_epoch_train_loss / train_n_batches,
                "Train acc": 100 * (train_correct / TRAIN_DATASET_SIZE),
                "Test acc": 100 * (test_correct / TEST_DATASET_SIZE),
            })

        save_model(net, model_type, start_time, epoch)

class CallNN(nn.Module):
    def __init__(self):
        super().__init__()

        in_features = 4515
        self.l1 = nn.Sequential(
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Linear(in_features*2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.l1(x)

def create_riichi_model(device, train_dataloader, test_dataloader, start_time=int(time.time())):
    net = RiichiNN().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    train_n_batches = math.ceil(TRAIN_DATASET_SIZE / BATCH_SIZE)

    for epoch in tqdm(range(1, EPOCHS + 1)):
        total_epoch_train_loss = 0
        train_correct = 0
        for x, y in train_dataloader:
            X_train, y_train = x.to(device), y.to(device)
            y_pred = net(X_train)

            loss = criterion(y_pred, y_train)
            total_epoch_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == EPOCHS:
                predictions = (y_pred > 0.5).float()*1
                train_correct += (predictions == y_train).float().sum()

        if epoch % 10 == 0 or epoch == EPOCHS:
            test_correct = 0
            for x, y in test_dataloader:
                X_test, y_test = x.to(device), y.to(device)
                y_pred = net(X_test)

                predictions = (y_pred > 0.5).float()*1
                test_correct += (predictions == y_test).float().sum()

            if USE_WANDB: wandb.log({
                "Epoch": epoch,
                "Train loss": total_epoch_train_loss / train_n_batches,
                "Train acc": 100 * (train_correct / TRAIN_DATASET_SIZE),
                "Test acc": 100 * (test_correct / TEST_DATASET_SIZE),
            })

        save_model(net, "riichi", start_time, epoch)

class RiichiNN(nn.Module):
    def __init__(self):
        super().__init__()

        in_features = 4515
        self.l1 = nn.Sequential(
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Linear(in_features*2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.l1(x)

def create_discard_model(device, train_dataloader, test_dataloader, start_time=int(time.time())):
    net = DiscardNN().to(device)
    criterion = nn.CrossEntropyLoss()

    train_n_batches = math.ceil(TRAIN_DATASET_SIZE / BATCH_SIZE)

    for epoch in tqdm(range(1, EPOCHS + 1)):
        total_epoch_train_loss = 0
        train_correct = 0
        for x, y in train_dataloader:
            X_train, y_train = x.to(device), y.squeeze().to(device)
            y_pred = net(X_train)

            loss = criterion(y_pred, y_train)
            total_epoch_train_loss += loss
            net.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in net.parameters():
                    param -= LEARNING_RATE * param.grad

            if epoch % 10 == 0 or epoch == EPOCHS:
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
                "Train acc": 100 * (train_correct / TRAIN_DATASET_SIZE),
                "Test acc": 100 * (test_correct / TEST_DATASET_SIZE),
            })

        save_model(net, "discard", start_time, epoch)

class DiscardNN(nn.Module):
    def __init__(self):
        super().__init__()

        in_features = 4515
        self.l1 = nn.Sequential(
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Linear(in_features*2, 34),
        )

    def forward(self, x):
        return self.l1(x)

def get_dataloaders(model_type, shuffle=True):
    total_size = TRAIN_DATASET_SIZE + TEST_DATASET_SIZE
    match model_type:
        case "chii": dataset = ChiiDataset(total_size)
        case "pon": dataset = PonDataset(total_size)
        case "kan": dataset = KanDataset(total_size)
        case "riichi": dataset = RiichiDataset(total_size)
        case "discard": dataset = DiscardDataset(total_size)
    train_dataset, test_dataset = random_split(dataset, (TRAIN_DATASET_SIZE, TEST_DATASET_SIZE))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    print(f"Train size: {len(train_dataset)}, test size: {len(test_dataset)}")

    return train_dataloader, test_dataloader

def save_model(net, model_type, timestamp, current_epoch):
    if SAVE_INTERVAL and (current_epoch % SAVE_INTERVAL == 0 or current_epoch == EPOCHS):
        fn = f"{SAVE_PATH}/{model_type}_{timestamp}_{current_epoch}.pt"
        torch.save(net.state_dict(), fn)
        print(f"Saved model {fn} with size {os.path.getsize(fn)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--model", choices=("chii", "pon", "kan", "riichi", "discard"))

    args = parser.parse_args()

    main(args.model)
