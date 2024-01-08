import torch
from torch.utils.data import Dataset, DataLoader
import sys
import pickle
import numpy as np

from mlops_ehaa.data.dataload import mnist


def predict(model: torch.nn.Module, dataloader: DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch["data"]) for batch in dataloader], 0)


class PredSet(Dataset):
    def __init__(self, data, device="cpu"):
        self.device = device
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": torch.Tensor(self.data[idx])
            .view((1, self.data[idx].shape[0], self.data[idx].shape[1]))
            .to(self.device),
        }


if __name__ == "__main__":
    assert (
        len(sys.argv) == 3
    ), "CL arguments should include a path to model checkpoint and a path to data set (either .npy or pickle)"

    model_pth = sys.argv[1]
    data_pth = sys.argv[2]
    try:
        data = pickle.load(data_pth)
    except:
        data = np.load(data_pth)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = DataLoader(PredSet(data, device=device), 1)
    model = torch.load(model_pth).to(device)
    preds = predict(model, dl)

    ps = torch.exp(preds).cpu()
    _, top_classes = ps.topk(1, dim=1)

    print(top_classes.flatten())
