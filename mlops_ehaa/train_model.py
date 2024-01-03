import sys

import torch
from models.model import MyAwesomeModel
import matplotlib.pyplot as plt

from data.dataload import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID = 0


def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    EPOCHS = 1
    model = MyAwesomeModel().to(device)
    train_set, _ = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = list()

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_set):
            x = batch["data"]
            y = batch["label"]

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.cpu().detach().numpy())

    torch.save(model, f"model/model_{ID}.pth")
    plt.plot(train_loss)
    plt.show()


if __name__ == "__main__":
    train(sys.argv[1])
