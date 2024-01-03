import torch
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class
class MNISTDataset(Dataset):
    def __init__(self, data_path, device="cpu"):
        self.device = device

        data_dict = torch.load(data_path)
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


def mnist(data_path: None | list[str, str] = None):
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not data_path:
        trainloader = MNISTDataset(data_path="data/processed/corruptmnist/train_set.pth", device=device)
        testloader = MNISTDataset(data_path="data/processed/corruptmnist/test_set.pth", device=device)

    trainset = DataLoader(trainloader, batch_size, shuffle=True)
    testset = DataLoader(testloader, batch_size, shuffle=True)

    return trainset, testset


if __name__ == "__main__":
    train, _ = mnist()
    for batch in train:
        print(batch["data"].shape)
        break
