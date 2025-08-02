from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.entireid import ENTIReID
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset


ENTIREID_DIR_PATH = DATA_DIR_PATH / 'reid' / 'entireid_blurred'
assert ENTIREID_DIR_PATH.exists()


def test_entireid_dataset():
    entireid_ds = ENTIReID(ENTIREID_DIR_PATH)
    assert len(entireid_ds) > 0

    assert len(entireid_ds._labels) > 0

    assert entireid_ds.classes == [*range(2741)]
    assert entireid_ds.class_count == 2741

    X, y = entireid_ds[0]
    assert X is not None
    print(X.shape)
    assert y == 0
    assert int(entireid_ds.image_class_name(0)) == y

    X, y = entireid_ds[-1]
    print(X.shape)
    assert X is not None
    assert y == 2740
    assert int(entireid_ds.image_class_name(-1)) == y


def test_triplet_database_using_entireid():
    entireid_ds = ENTIReID(ENTIREID_DIR_PATH)
    entireid_tds = TripletDataset(entireid_ds)

    n = len(entireid_tds)
    assert n != 0

    # expensive and unnecessary.
    anchors = [triplet[0] for triplet in entireid_tds.triplets]
    positives = [triplet[1] for triplet in entireid_tds.triplets]
    negatives = [triplet[2] for triplet in entireid_tds.triplets]

    for (a, p, n) in zip(anchors, positives, negatives):
        ya = entireid_ds.image_class_ids[a]
        yp = entireid_ds.image_class_ids[p]
        yn = entireid_ds.image_class_ids[n]

        assert ya == yp
        assert ya != yn
        assert yp != yn
