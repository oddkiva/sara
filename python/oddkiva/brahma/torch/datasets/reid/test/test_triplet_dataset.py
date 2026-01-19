# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import (
    TripletDataset)
from oddkiva.brahma.torch.datasets.reid.eth123 import ETH123


def test_triplet_sampling():
    eth123_root_path = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
    assert eth123_root_path.exists()
    eth123_ds = ETH123(eth123_root_path)
    eth123_tds = TripletDataset(eth123_ds)

    n = len(eth123_tds)
    assert n != 0

    max_samples = 10

    print("Checking all ETH123 triplet samples...")
    for i, sample in enumerate(eth123_tds):
        if i > max_samples:
            break
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
