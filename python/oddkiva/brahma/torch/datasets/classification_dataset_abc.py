from typing import List
from abc import abstractmethod

from torch.utils.data import Dataset


class ClassificationDatasetABC(Dataset):

    @property
    @abstractmethod
    def classes(self):
        raise NotImplementedError(
            'Subclasses of ClassificationDataset must implement the '
            '`classes` property'
        )

    @property
    def class_count(self):
        return len(self.classes)

    @property
    @abstractmethod
    def image_class_ids(self) -> List[int]:
        raise NotImplementedError((
            'Subclasses of ClassificationDataset must implement '
            'the `image_class_ids` property'
        ))

    @abstractmethod
    def image_class_name(self, idx: int) -> str:
        raise NotImplementedError((
            'Subclasses of ClassificationDataset must implement '
            '`image_class_name` method'
        ))
