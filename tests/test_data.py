from mlops_ehaa.data.dataload import mnist


def test_mnist():
    bt_size = 8
    out = mnist(batch_size=bt_size)

    assert len(out) == 2
