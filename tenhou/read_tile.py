import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import sys

models_dir = "models"
image_dir_root = "data/tiles"
image_dir_private = image_dir_root + "/private"
image_dir_discard = image_dir_root + "/discard"
image_dir_discard_v = image_dir_root + "/discard_v"
image_dir_discard_h = image_dir_root + "/discard_h"
resized_dir = image_dir_root + "/resized"
in_width = 16
in_height = round(in_width * 1.5)
tiles = [
    "0m", "1m", "2m", "3m", "4m", # Char 1-4 + red
    "5m", "6m", "7m", "8m", "9m", # Char 5-9
    "0p", "1p", "2p", "3p", "4p", # Dots 1-4 + red
    "5p", "6p", "7p", "8p", "9p", # Dots 5-9
    "0s", "1s", "2s", "3s", "4s", # Bamb 1-4 + red
    "5s", "6s", "7s", "8s", "9s", # Bamb 5-9
    "ew", "sw", "ww", "nw",       # Winds
    "wd", "gd", "rd"              # Dragons
]
num_tiles = len(tiles)

def main():
    create_private_tile_model()
    print("Finished private tile model")
    create_discard_tile_model()
    print("Finished discard tile model")

def create_private_tile_model():
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    model_path = os.path.join(models_dir, "private_tile_recognition")
    if os.path.exists(model_path):
        model = TileNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    files = [f for f in os.listdir(image_dir_private) if os.path.isfile(os.path.join(image_dir_private, f))]
    preprocess_all(files, image_dir_private)
    model = create_model(files)
    torch.save(model.state_dict(), model_path)
    return model

def create_discard_tile_model():
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    model_path = os.path.join(models_dir, "discard_tile_recognition")
    if os.path.exists(model_path):
        model = TileNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    files = [f for f in os.listdir(image_dir_discard) if os.path.isfile(os.path.join(image_dir_discard, f))]
    preprocess_all(files, image_dir_discard)
    model = create_model(files)
    torch.save(model.state_dict(), model_path)
    return model

def create_discard_tile_models():
    files = [f for f in os.listdir(image_dir_discard_h) if os.path.isfile(os.path.join(image_dir_discard_h, f))]
    preprocess_all(files, image_dir_discard_h)
    model_h = create_model(files)

    files = [f for f in os.listdir(image_dir_discard_v) if os.path.isfile(os.path.join(image_dir_discard_v, f))]
    preprocess_all(files, image_dir_discard_v)
    model_v = create_model(files)

    return model_h, model_v

def create_model(files):
    X_train = [cv2.imread(os.path.join(resized_dir, f), cv2.IMREAD_COLOR) for f in files]
    X_train = torch.from_numpy(np.array([np.array(i).flatten().astype(np.float32) for i in X_train]))
    v_min, v_max = X_train.min(), X_train.max()
    X_train = (X_train - v_min) / (v_max - v_min)

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
    failed = False
    for i in range(len(y_pred)):
        if y_pred[i] != labels[i]:
            print(files[i], "Predicted", tiles[int(y_pred[i])], "Expected", tiles[int(labels[i])])
            failed = True

    if failed:
        print("Tile recognition model was imperfect!")
        #sys.exit(-1)

    return model

def predict(model, X):
    single = False
    if type(X) == "int":
        single = True
        X = np.array([X])

    X = torch.from_numpy(np.array([np.array(i).flatten().astype(np.float32) for i in X]))

    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    labels = [tiles[i] for i in y_pred]

    if single:
        return labels[0]
    else:
        return labels

class TileNN(nn.Module):
    def __init__(self):
        super(TileNN, self).__init__()
        features = in_width * in_height * 3
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(features, round(features*1.5)),
            nn.ReLU(),
            nn.Linear(round(features*1.5), round(features*1.5)),
            nn.ReLU(),
            nn.Linear(round(features*1.5), num_tiles)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def preprocess(tile_img, min_value=160):
    tile_img = cv2.resize(tile_img, (in_width, in_height))
    return tile_img
    for y in range(tile_img.shape[0]):
        for x in range(tile_img.shape[1]):
            if tile_img[y, x] >= min_value:
                tile_img[y, x] = 255
    for y in range(tile_img.shape[0]):
        if tile_img[y, 0] < min_value:
            tile_img[y, 0] = 255
        if tile_img[y, tile_img.shape[1]-1] < min_value:
            tile_img[y, tile_img.shape[1]-1] = 255
    for x in range(tile_img.shape[1]):
        if tile_img[0, x] < min_value:
            tile_img[0, x] = 255
        if tile_img[tile_img.shape[0]-1, x] < min_value:
            tile_img[tile_img.shape[0]-1, x] = 255
    crop_from_x = 0
    crop_from_y = 0
    crop_to_x = tile_img.shape[1] - 1
    crop_to_y = tile_img.shape[0] - 1
    while crop_from_x < crop_to_x:
        break_outer = False
        for y in range(tile_img.shape[0]):
            if tile_img[y, crop_from_x] < min_value:
                break_outer = True
                break
        if break_outer:
            break
        crop_from_x += 1
    while crop_from_x < crop_to_x:
        break_outer = False
        for y in range(tile_img.shape[0]):
            if tile_img[y, crop_to_x] < min_value:
                break_outer = True
                break
        if break_outer:
            break
        crop_to_x -= 1
    while crop_from_y < crop_to_y:
        break_outer = False
        for x in range(tile_img.shape[1]):
            if tile_img[crop_from_y, x] < min_value:
                break_outer = True
                break
        if break_outer:
            break
        crop_from_y += 1
    while crop_from_y < crop_to_y:
        break_outer = False
        for x in range(tile_img.shape[1]):
            if tile_img[crop_to_y, x] < min_value:
                break_outer = True
                break
        if break_outer:
            break
        crop_to_y -= 1
    if crop_from_x != crop_to_x and crop_from_y != crop_to_y:
        tile_img = tile_img[crop_from_y:crop_to_y, crop_from_x:crop_to_x]
    tile_img = cv2.resize(tile_img, (in_width, in_height))
    return tile_img

def preprocess_all(files, image_dir):
    for f in files:
        file_path = os.path.join(image_dir, f)
        new_file_path = os.path.join(resized_dir, f)
        if not os.path.exists(new_file_path):
            tile_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            preprocessed = preprocess(tile_img)
            cv2.imwrite(new_file_path, preprocessed)

def get_label(filename):
    for i, tile in enumerate(tiles):
        if tile in filename:
            return i
    print("Failed to find", filename, "label")
    sys.exit(-1)

if __name__ == "__main__":
    main()