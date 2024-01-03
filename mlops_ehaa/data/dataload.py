import torch
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class
class MNISTDataset(Dataset):
    def __init__(self, test=False, device="cpu"):
        self.device = device
        if test:
            data_dict = torch.load("data/processed/corruptmnist/test_set.pth")
            self.data = data_dict["data"]
            self.labels = data_dict["labels"]
        else:
            data_dict = torch.load("data/processed/corruptmnist/train_set.pth")
            self.data = data_dict["data"]
            self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "data": torch.Tensor(self.data[idx])
            .view((1, self.data[idx].shape[0], self.data[idx].shape[1]))
            .to(self.device),
            "label": torch.Tensor(self.labels[idx]).to(self.device),
        }
        return sample


def mnist():
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader = MNISTDataset(test=False, device=device)
    testloader = MNISTDataset(test=True, device=device)

    trainset = DataLoader(trainloader, batch_size, shuffle=True)
    testset = DataLoader(testloader, batch_size, shuffle=True)

    return trainset, testset


if __name__ == "__main__":
    train, _ = mnist()
    for batch in train:
        print(batch["data"].shape)
        break
