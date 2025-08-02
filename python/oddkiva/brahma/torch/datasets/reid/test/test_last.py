from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.last import LaST
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset


LAST_DIR_PATH = DATA_DIR_PATH / 'reid' / 'last'
assert LAST_DIR_PATH.exists()

def test_last_dataset():
    last_ds = LaST(LAST_DIR_PATH)
    assert len(last_ds) > 0

    assert last_ds.class_count > 0

    X, y = last_ds[0]
    assert X is not None
    print(X.shape)
    assert type(y) is int

    last_tds = TripletDataset(last_ds)

    n = len(last_tds)
    assert n > 0

    # We simply check the labels as reloading every single images is very
    # expensive and unnecessary.
    anchors = [triplet[0] for triplet in last_tds.triplets]
    positives = [triplet[1] for triplet in last_tds.triplets]
    negatives = [triplet[2] for triplet in last_tds.triplets]

    for (a, p, n) in zip(anchors, positives, negatives):
        ya = last_tds.base_dataset.image_class_ids[a]
        yp = last_tds.base_dataset.image_class_ids[p]
        yn = last_tds.base_dataset.image_class_ids[n]

        assert ya == yp
        assert ya != yn
        assert yp != yn
