import os
import pytest
from mlops_ehaa.data.dataload import mnist


@pytest.mark.skipif(
    not os.path.exists("data/processed/corruptmnist/train_set.pth"),
    reason="Data files not found",
)
def test_mnist():
    bt_size = 8
    out = mnist(batch_size=bt_size)

    assert len(out) == 2
