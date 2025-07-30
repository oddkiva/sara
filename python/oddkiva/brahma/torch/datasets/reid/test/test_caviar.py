from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.caviar import CAVIAR
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDatabase


def test_triplet_sampling():
    caviar_dir_path = DATA_DIR_PATH / 'reid' / 'CAVIARa'
    assert caviar_dir_path.exists()

    caviar_ds = CAVIAR(caviar_dir_path)
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


    caviar_tds = TripletDatabase(caviar_ds)

    n = len(caviar_tds)
    assert n != 0

    (Xa, Xb, Xc), (ya, yb, yc) = caviar_tds[-1]
    assert Xa is not None
    assert Xb is not None
    assert Xc is not None

    assert type(ya) is int
    assert type(yb) is int
    assert type(yc) is int

    assert ya >= 0
    assert yb >= 0
    assert yc >= 0
