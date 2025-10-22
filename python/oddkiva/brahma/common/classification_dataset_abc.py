from abc import abstractmethod


class ClassificationDatasetABC:

    @property
    @abstractmethod
    def classes(self) -> list:
        ...

    @property
    def class_count(self):
        return len(self.classes)

    @property
    @abstractmethod
    def image_class_ids(self) -> list[int]:
        ...

    @abstractmethod
    def image_class_name(self, idx: int) -> str:
        ...
