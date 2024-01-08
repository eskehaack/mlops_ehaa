import torch

from mlops_ehaa.models.model import MyAwesomeModel


def test_MyAwesomeModel():
    model = MyAwesomeModel()
    a = torch.ones((1, 1, 28, 28))
    b = model(a)
    assert b.shape == (1, 10), "Model output in wrong shape"
