"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

from typing import Sequence

import random

from PIL import Image

import torch
import torch.nn.functional as TF
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.tv_tensors import (
    BoundingBoxes,
    BoundingBoxFormat
)

from oddkiva.brahma.torch.datasets.coco.dataset import (
    COCOObjectDetectionDataset
)


class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self,
                 output_size: int = 320,
                 max_size: int | None = None,
                 rotation_range: float = 0,
                 translation_range: tuple[float, float] = (0.1, 0.1),
                 scaling_range: tuple[float, float] = (0.5, 1.5),
                 probability: float = 1.0,
                 fill_value: float = 114,
                 use_cache: bool = True,
                 max_cached_images: int = 50,
                 random_pop: bool = True) -> None:
        """
        Parameters:
            output_size:
                Target size for resizing individual images.
            rotation_range:
                Range of rotation in degrees for affine transformation.
            translation_range:
                Range of translation for affine transformation.
            scaling_range:
                Range of scaling factors for affine transformation.
            probability:
                Probability of applying the Mosaic augmentation.
            fill_value:
                Fill value for padding or affine transformations.
            use_cache:
                Whether to use cache. Defaults to True.
            max_cached_images:
                The maximum length of the cache.
            random_pop:
                Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(
            degrees=rotation_range,
            translate=translation_range,
            scale=scaling_range,
            fill=fill_value
        )
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self,
                                  image_i: torch.Tensor,
                                  targets_i: dict[str, torch.Tensor],
                                  dataset: COCOObjectDetectionDataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image.
        get_size = F.get_size if hasattr(F, "get_size") else F.get_spatial_size

        # Fetch the first image.
        image_i, targets_i = self.resize(image_i, targets_i)
        resized_images, resized_targets = [image_i], [targets_i]
        h_max, w_max = get_size(resized_images[0])

        # Randomly select 3 images.
        annotation_count = len(dataset)
        image_ixs = random.choices(range(annotation_count), k=3)
        for i in image_ixs:
            image_i, boxes_i, labels_i = dataset.read_annotated_image(i)

            # Transform the annotations.
            targets_i = {
                'boxes': boxes_i,
                'labels': labels_i
            }
            image_i, targets_i = self.resize(image_i, targets_i)
            h_i, w_i = get_size(image_i)

            h_max, w_max = max(h_max, h_i), max(w_max, w_i)
            resized_images.append(image_i)
            resized_targets.append(targets_i)

        return resized_images, resized_targets, h_max, w_max

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                # Do not remove the last image
                index = random.randint(0, len(cache) - 2)
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(),
                               labels=self._clone(cache[idx]["labels"]))
                          for idx in sample_indices]  # sample 3 images
        mosaic_samples = [
            dict(img=image.copy(), labels=self._clone(target))
        ] + mosaic_samples

        get_size = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        image_sizes = [get_size(mosaic_samples[idx]["img"]) for idx in range(4)]
        h_max = max(size[0] for size in image_sizes)
        w_max = max(size[1] for size in image_sizes)

        return mosaic_samples, h_max, w_max

    def create_mosaic_from_cache(self, mosaic_samples, h: int, w: int):
        placement_offsets = [[0, 0],
                             [w, 0],
                             [0, h],
                             [w, h]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode,
                                 size=(w * 2, h * 2),
                                 color=0)
        offsets = torch.tensor([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])
        offsets = TF.pad(offsets, (0, 2, 0, 0), 'constant', 0)

        mosaic_target = []
        for i, sample_i in enumerate(mosaic_samples):
            img_i = sample_i["img"]
            target_i = sample_i["labels"]

            merged_image.paste(img_i, placement_offsets[i])
            target_i['boxes'] = target_i['boxes'] + offsets[i]
            mosaic_target.append(target_i)

        merged_target = {}
        for key in mosaic_target[0]:
            merged_target[key] = torch.cat([
                target[key] for target in mosaic_target
            ])

        return merged_image, merged_target

    def create_mosaic_from_dataset(self,
                                   images,
                                   targets: dict[str, torch.Tensor],
                                   h: int, w: int):
        """Creates a mosaic image by combining multiple images."""

        placement_offsets = [[0, 0], [w, 0], [0, h], [w, h]]
        merged_image = Image.new(mode='RGB',
                                 size=(w * 2, h * 2),
                                 color=0)
        for i, img in enumerate(images):
            pil_img = F.to_pil_image(img)
            merged_image.paste(pil_img, placement_offsets[i])

        # Merges targets into a single target dictionary for the mosaic.
        offsets = torch.tensor([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])
        offsets = TF.pad(offsets, (0, 2, 0, 0), 'constant', 0)

        merged_targets = {}
        for key in targets[0]:
            if key == 'boxes':
                assert targets[0]['boxes'].
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            if isinstance(values[0], torch.Tensor):
                merged_targets[key] = torch.cat(values, dim=0)
            else:
                merged_targets[key] = values

        return merged_image, merged_targets

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Parameters:
            inputs:
                The tuple (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, targets, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, targets, dataset

        # Prepare mosaic components
        if self.use_cache:
            (mosaic_samples,
             max_height,
             max_width) = self.load_samples_from_cache(image,
                                                       targets,
                                                       self.mosaic_cache)
            mosaic_image, mosaic_targets = self.create_mosaic_from_cache(
                mosaic_samples,
                max_height,
                max_width
            )
        else:
            (resized_images,
             resized_targets,
             max_height,
             max_width) = self.load_samples_from_dataset(image, targets, dataset)
            mosaic_image, mosaic_targets = self.create_mosaic_from_dataset(
                resized_images,
                resized_targets,
                max_height,
                max_width
            )

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_targets:
            mosaic_targets['boxes'] = BoundingBoxes(
                mosaic_targets['boxes'],
                format=BoundingBoxFormat.XYXY,
                canvas_size=mosaic_image.size[::-1]
            )

        # Apply affine transformations
        mosaic_image, mosaic_targets = self.affine_transform(
            mosaic_image,
            mosaic_targets
        )

        return mosaic_image, mosaic_targets, dataset

