from typing import List
from abc import abstractmethod

from torch.utils.data import Dataset


class ClassificationDatasetABC(Dataset):

    @property
    @abstractmethod
    def classes(self) -> List:
        ...

    @property
    def class_count(self):
        return len(self.classes)

    @property
    @abstractmethod
    def image_class_ids(self) -> List[int]:
        ...

    @abstractmethod
    def image_class_name(self, idx: int) -> str:
        ...
