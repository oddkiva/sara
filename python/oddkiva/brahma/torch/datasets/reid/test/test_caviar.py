from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.caviar import CAVIAR
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset


CAVIAR_DIR_PATH = DATA_DIR_PATH / 'reid' / 'CAVIARa'
assert CAVIAR_DIR_PATH.exists()


def test_caviar_dataset():
    caviar_ds = CAVIAR(CAVIAR_DIR_PATH)
    assert len(caviar_ds) > 0

    assert len(caviar_ds._labels) > 0

    assert caviar_ds.classes == [*range(72)]
    assert caviar_ds.class_count == 72

    X, y = caviar_ds[0]
    assert X is not None
    print(X.shape)
    assert y == 0
    assert int(caviar_ds.image_class_name(0)) - 1 == y

    X, y = caviar_ds[10]
    assert X is not None
    print(X.shape)
    assert y == 1
    assert int(caviar_ds.image_class_name(10)) - 1 == y

    X, y = caviar_ds[-1]
    print(X.shape)
    assert X is not None
    assert y == 71
    assert int(caviar_ds.image_class_name(-1)) - 1 == y


def test_triplet_database_using_caviar():
    caviar_ds = CAVIAR(CAVIAR_DIR_PATH)
    caviar_tds = TripletDataset(caviar_ds)

    n = len(caviar_tds)
    assert n != 0

    for sample in caviar_tds:
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
        assert yp != yn
