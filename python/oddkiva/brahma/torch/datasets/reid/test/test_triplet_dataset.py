from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import (
    TripletDatabase)
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123


def test_triplet_sampling():
    eth123_root_path = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ/'
    assert eth123_root_path.exists()
    eth123_ds = ETH123(eth123_root_path)

    eth123_tds = TripletDatabase(eth123_ds)

    n = len(eth123_tds)
    assert n != 0

    (Xa, Xb, Xc), (ya, yb, yc) = eth123_tds[-1]
    assert Xa is not None
    assert Xb is not None
    assert Xc is not None

    assert type(ya) is int
    assert type(yb) is int
    assert type(yc) is int

    assert ya >= 0
    assert yb >= 0
    assert yc >= 0
