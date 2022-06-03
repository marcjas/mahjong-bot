from datasets import DiscardDataset
from torch.utils.data import DataLoader 

def main():
    dataset = DiscardDataset()
    dataloader =  DataLoader(dataset, batch_size=50)

    train_features, train_labels = next(iter(dataloader))



if __name__ == "__main__":
    main()



