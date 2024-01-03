import torch
from torch.utils.data import Dataset, DataLoader


# Define a custom dataset class
class MNISTDataset(Dataset):
    def __init__(self, test=False, device="cpu"):
        train = torch.tensor([])
        train_labels = torch.tensor([], dtype=torch.long)

        if not test:
            for i in range(6):
                train = torch.cat(
                    (train, torch.load(f"data/corruptmnist/train_images_{i}.pt"))
                )
                train_labels = torch.cat(
                    (train_labels, torch.load(f"data/corruptmnist/train_target_{i}.pt"))
                )
                self.data = train.view(-1, 784).to(device)
                self.labels = train_labels.flatten().to(device)

        else:
            test = torch.load(f"data/corruptmnist/test_images.pt")
            test_labels = torch.load(f"data/corruptmnist/test_target.pt").long()
            self.data = test.view(-1, 784).to(device)
            self.labels = test_labels.flatten().to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "data": torch.Tensor(self.data[idx]),
            "label": torch.Tensor(self.labels[idx]),
        }
        return sample


def mnist():
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

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
