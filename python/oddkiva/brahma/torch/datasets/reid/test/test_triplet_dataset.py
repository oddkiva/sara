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

    print("Checking all ETH123 triplet samples...")
    for sample in eth123_tds:
        (Xa, Xp, Xn), (ya, yp, yn) = sample
        assert Xa is not None
        assert Xp is not None
        assert Xn is not None

        assert type(ya) is int
        assert type(yp) is int
        assert type(yn) is int

        assert ya >= 0
        assert yp >= 0
        assert yn >= 0

        assert ya == yp
        assert ya != yn
