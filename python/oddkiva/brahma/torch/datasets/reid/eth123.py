import os
from pathlib import Path
from typing import List
from typing_extensions import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io.image import decode_image


class ETH123(Dataset):

    def __init__(self, root_path: Path):
        self._root_path = root_path

        # Populate the list of class directory paths
        paths = [
            Path(x[0]).relative_to(root_path)
            for x in os.walk(root_path)
        ]
        self._class_paths = [
            p for p in paths
            if str(p) not in [f'seq{i}' for i in [1, 2, 3]]
            and str(p) != '.'
        ]

        # Populate the list of image filenames for each class.
        self._image_filenames_per_class = [
            os.listdir(root_path / p) for p in self._class_paths
        ]

        # Transform the list of file names into a list of file paths.
        self._image_paths_per_class: List[List[Path]] = []
        # fns = filenames
        for image_fns, class_path in zip(self._image_filenames_per_class,
                                         self._class_paths):
            # fp = filepath
            image_fps = [root_path / class_path / fn for fn in image_fns]
            self._image_paths_per_class.append(image_fps)

            image_fps_exist = all([fp.exists() for fp in image_fps])
            if not image_fps_exist:
                raise ValueError('OOOOPS')

        self._classes = [str(p) for p in self._class_paths]
        self._classes.sort()

        self._image_paths = [
            p
            for ps in self._image_paths_per_class
            for p in ps
        ]

        def image_class(p: Path) -> str:
            return '/'.join(
                str(p.relative_to(root_path)).split('/')[:-1]
            )
        self._image_labels = [
            image_class(p) for p in self._image_paths
        ]

        self._labels = list(set(self._image_labels))
        self._labels.sort()

        self._image_label_ixs = [
            self._labels.index(image_label)
            for image_label in self._image_labels
        ]

    def __len__(self):
        return len(self._image_paths)


    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        return decode_image(self._image_paths[idx]), self._image_label_ixs[idx]


if __name__ == '__main__':
    ds = ETH123('/Users/oddkiva/Downloads/reid/dataset_ETHZ/')
    import IPython; IPython.embed()
