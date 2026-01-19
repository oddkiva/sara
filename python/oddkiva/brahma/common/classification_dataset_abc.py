# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from abc import ABC, abstractmethod
from typing import Any


class ClassificationDatasetABC(ABC):
    """Abstract Base Classification Dataset class
    """

    @property
    @abstractmethod
    def classes(self) -> list:
        """Returns the list of classes.
        """
        ...

    @property
    def class_count(self) -> int:
        """Returns the number of classes in the dataset.
        """
        return len(self.classes)

    @property
    @abstractmethod
    def image_class_ids(self) -> list[int]:
        r"""
        Returns the mapping $i \mapsto l_i$ where $l_i$ is the class ID (i.e.,
        the label) of image indexed by $i$.
        """
        ...

    @abstractmethod
    def image_class_name(self, i: int) -> str:
        r"""
        Returns the class ID (i.e., the label) $l_i$ of the image indexed by
        $i$.
        """
        ...

    @abstractmethod
    def __getitem__(self, i: int) -> tuple[Any, Any]:
        r"""
        Returns the image-label pair $(\mathbf{I}_i, l_i) \in \mathbb{R}^{3
        \times H \times W} \times \left\{ 0, \dots, L - 1 \right\}$.
        """
        ...
