from datasets import DiscardDataset
from torch.utils.data import DataLoader 
import torch
import torch.nn as nn
from tqdm import tqdm

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

BATCH_SIZE = 1000

def main():
    dataset = DiscardDataset()
    dataloader =  DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = next(iter(dataloader))

    learning_rate = 1e-1
    loss_list = []
    for _ in tqdm(range(10)):
        y_pred = net(X_train)
        loss = criterion(y_pred, y_train)
        loss_list.append(loss.item())
        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in net.parameters():
                param -= learning_rate * param.grad

    logits = net(X_train)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(len(y_pred))
    print(y_pred[0])
    wrong_cnt = 0
    for i in range(len(y_pred)):
        label = tiles[(y_train[i] == 1).nonzero(as_tuple=True)[0]]
        predicted_tile = tiles[y_pred[i]]
        print(f"predicted: {predicted_tile}, actual discard: {label}")
        if label != predicted_tile:
            wrong_cnt += 1
    accuracy = 100 - (100*(wrong_cnt / len(y_train)))
    print(f"accuracy: {accuracy}%")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4923, 3000),
            nn.ReLU(),
            nn.Linear(3000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 34),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__":
    main()


