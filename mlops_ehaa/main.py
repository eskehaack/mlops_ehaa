import click
import torch
from model import MyAwesomeModel
import matplotlib.pyplot as plt
from numpy import mean

from data import mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
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

    torch.save(model, "checkpoint.pth")
    plt.plot(train_loss)
    plt.show()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint).to(device)
    _, test_set = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = list()
    equals = list()

    for i, batch in enumerate(test_set):
        x = batch["data"]
        y = batch["label"]
        pred = model(x)

        loss = criterion(pred, y)
        loss.backward()

        ps = torch.exp(pred).cpu()
        _, top_class = ps.topk(1, dim=1)
        equals.extend(top_class == y.view(top_class.shape).type(torch.FloatTensor))

        test_loss.append(loss.cpu().detach().numpy())

    else:
        accuracy = mean(equals)
        print(accuracy)


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    print(device)
    cli()
