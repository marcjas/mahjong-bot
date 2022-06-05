from datasets import DiscardDataset
from torch.utils.data import DataLoader 
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import wandb

wandb.init(project="riichi-mahjong", entity="shuthus")

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

DATASET_SIZE = 10000 # set to None if you want to load everything
BATCH_SIZE = 5000
SAVE_INTERVAL = 200 

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

    learning_rate = 0.1
    epochs = 1000

    wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": BATCH_SIZE 
    }

    for epoch in tqdm(range(epochs)):
        loss_list = []
        for x, y in dataloader:
            X_train = x.to(device)
            y_train = y.to(device)
            y_pred = net(X_train)
            loss = criterion(y_pred, y_train)
            loss_list.append(loss.item())

        wandb.log({"loss": loss_list[-1]})
        net.zero_grad() # clear gradients of model parameters
        loss.backward() # update weights
        with torch.no_grad():
            for param in net.parameters():
                param -= learning_rate * param.grad

        if epoch % SAVE_INTERVAL == SAVE_INTERVAL-1:
            torch.save(net.state_dict(), f"D:/models/discard_model_{epoch}")

    logits = net(X_train)
    pred_prob = nn.Softmax(dim=1)(logits)
    y_pred = torch.argmax(pred_prob, dim=1)

    wrong_cnt = 0
    for i in range(len(y_pred)):
        label = tiles[(y_train[i] == 1).nonzero(as_tuple=True)[0]]
        predicted_tile = tiles[y_pred[i]]
        if label != predicted_tile:
            wrong_cnt += 1
    accuracy = 100 - (100*(wrong_cnt / len(y_train)))
    print(f"accuracy: {accuracy}%")


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
        # x = x.view(1000, 4923, 1)
        # x = self.c1(x)
        return self.l1(x)

if __name__ == "__main__":
    main()


