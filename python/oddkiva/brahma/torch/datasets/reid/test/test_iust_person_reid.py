# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.datasets.reid.iust_person_reid import IUSTPersonReID
from oddkiva.brahma.torch.datasets.reid.triplet_dataset import TripletDataset


IUST_PERSON_REID_DIR_PATH = DATA_DIR_PATH / 'reid' / 'IUSTPersonReID'
assert IUST_PERSON_REID_DIR_PATH.exists()


def test_iust_train_dataset():
    iust_ds = IUSTPersonReID(IUST_PERSON_REID_DIR_PATH, dataset_type='train')
    assert len(iust_ds) > 0

    assert len(iust_ds.image_class_ids) > 0

    assert iust_ds.class_count > 0

    X, y = iust_ds[0]
    assert X is not None
    print(X.shape)
    print(y)

    X, y = iust_ds[-1]
    print(X.shape)
    print(y)

def test_iust_test_dataset():
    iust_ds = IUSTPersonReID(IUST_PERSON_REID_DIR_PATH, dataset_type='test')
    assert len(iust_ds) > 0
    assert len(iust_ds.image_class_ids) > 0

    assert iust_ds.class_count > 0

    X, y = iust_ds[0]
    assert X is not None
    print(X.shape)
    print(y)

    X, y = iust_ds[-1]
    print(X.shape)
    print(y)


def test_triplet_database_using_iust():
    iust_ds = IUSTPersonReID(IUST_PERSON_REID_DIR_PATH)
    iust_tds = TripletDataset(iust_ds)

    n = len(iust_tds)
    assert n != 0

    # expensive and unnecessary.
    anchors = [triplet[0] for triplet in iust_tds.triplets]
    positives = [triplet[1] for triplet in iust_tds.triplets]
    negatives = [triplet[2] for triplet in iust_tds.triplets]

    for (a, p, n) in zip(anchors, positives, negatives):
        ya = iust_ds.image_class_ids[a]
        yp = iust_ds.image_class_ids[p]
        yn = iust_ds.image_class_ids[n]

        assert ya == yp
        assert ya != yn
        assert yp != yn
