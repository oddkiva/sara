# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from abc import ABC, abstractmethod
from typing import Any


class ClassificationDatasetABC(ABC):

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

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        ...
