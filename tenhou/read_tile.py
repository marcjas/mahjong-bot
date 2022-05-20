import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import sys

image_dir = "data/tiles"
resized_dir = image_dir + "/resized"
in_width = 8
in_height = int(in_width * 1.5)
tiles = [
    "0m", "1m", "2m", "3m", "4m", # Char 1-4 + red
    "5m", "6m", "7m", "8m", "9m", # Char 5-9
    "0p", "1p", "2p", "3p", "4p", # Dots 1-4 + red
    "5p", "6p", "7p", "8p", "9p", # Dots 5-9
    "0s", "1s", "2s", "3s", "4s", # Bamb 1-4 + red
    "5s", "6s", "7s", "8s", "9s", # Bamb 5-9
    "1z", "2z", "3z", "4z",       # Winds
    "5z", "6z", "7z"              # Dragons
]
num_tiles = len(tiles)

def main():
    files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    resize_all(files)

    X_train = [cv2.imread(os.path.join(resized_dir, f), cv2.IMREAD_GRAYSCALE) for f in files]
    X_train = torch.from_numpy(np.array([np.array(i).flatten().astype(np.float32) for i in X_train]))
    v_min, v_max = X_train.min(), X_train.max()
    X_train = (X_train - v_min) / (v_max - v_min) * (1 - 0) + 0

    labels = torch.from_numpy(np.array([get_label(f) for f in files]).astype(np.longlong))
    
    model = TileNN().to("cpu")

    criterion = nn.CrossEntropyLoss()

    learning_rate = 1e-1
    loss_list = []
    for _ in range(1000):
        y_pred = model(X_train)
        loss = criterion(y_pred, labels)
        loss_list.append(loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    logits = model(X_train)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    for i in range(len(y_pred)):
        if y_pred[i] != labels[i]:
            print("Tile recognition model was imperfect!")
            sys.exit(-1)

    return model

def create_model():
    return main()

def predict(model, X):
    if type(X) == "int":
        X = np.array([X])

    X = torch.from_numpy(np.array([np.array(i).flatten().astype(np.float32) for i in X]))

    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    labels = [tiles[i] for i in y_pred]

    if len(labels) == 1:
        return labels[0]
    else:
        return labels

class TileNN(nn.Module):
    def __init__(self):
        super(TileNN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_width*in_height, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_tiles)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def resize_all(files):
    for f in files:
        file_path = os.path.join(image_dir, f)
        new_file_path = os.path.join(resized_dir, f)
        if not os.path.exists(new_file_path):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img, (in_width, in_height))
            cv2.imwrite(new_file_path, resized)

def get_label(filename):
    for i, tile in enumerate(tiles):
        if tile in filename:
            return i

if __name__ == "__main__":
    main()