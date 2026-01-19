# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.tf.datasets.reid.eth123 import ETH123, ETH123_Internal

def test_eth123_internal():
    ds_root_path = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
    assert ds_root_path.exists()
    ds = ETH123_Internal(ds_root_path)

    im, _ = ds[0]
    print(im)


def test_eth123():
    ds_root_path = DATA_DIR_PATH / 'reid' / 'dataset_ETHZ'
    assert ds_root_path.exists()
    ds = ETH123(ds_root_path)

    for e in ds.range(2):
        print(e)
