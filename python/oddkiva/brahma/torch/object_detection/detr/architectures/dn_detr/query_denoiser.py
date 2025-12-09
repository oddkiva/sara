from typing import Any
from dataclasses import dataclass

import torch
import torch.nn as nn


class BoxGeometryNoiser(nn.Module):
    """
    We follow the implementation of DN-DETR
    """

    @dataclass
    class Output:
        box_labels: torch.Tensor
        box_geometries: torch.Tensor
        attention_mask: torch.Tensor
        dn_meta: dict[str, Any]

    def __init__(self,
                 class_count: int,
                 box_count: int = 100,
                 box_noise_scale: float = 1.0):
        super().__init__()

        self.object_class_embedding_fn: nn.Module
        self.class_count = class_count
        self.box_count = box_count
        self.box_noise_scale = box_noise_scale

    def stack_box_annotations(
        self,
        box_annotations: dict[str, list[torch.Tensor]],
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = box_annotations['boxes']
        labels = box_annotations['labels']
        assert len(boxes) == len(labels)

        # What we do is simply tensorize the ground truth data,
        # with appropriate padding.
        box_count_max = max([len(b) for b in boxes])
        batch_size = len(boxes)

        stacked_box_labels = torch.full(
            [batch_size, box_count_max],
            num_classes,
            dtype=torch.int32,
            device=boxes[0].device
        )
        stacked_box_geometries = torch.zeros(
            [batch_size, box_count_max, 4],
            device=boxes[0].device
        )
        mask = torch.zeros([batch_size, box_count_max],
                           dtype=torch.bool,
                           device=boxes[0].device)

        for n in range(batch_size):
            box_count = len(boxes[n])
            # Let us impose this hard-constraint for now
            assert box_count > 0

            stacked_box_labels[n, :box_count] = labels[n]
            stacked_box_geometries[n, :box_count] = boxes[n]
            mask[n, :box_count] = True

        return stacked_box_labels, stacked_box_geometries, mask

    def forward(
        self,
        ground_truth: dict[str, list[torch.Tensor]]
    ) -> Output:
        boxes = ground_truth['boxes']
        labels = ground_truth['labels']
        assert len(boxes) == len(labels)

        box_count_max = max([len(b) for b in boxes])

        group_count = self.box_count // box_count_max
        if group_count == 0:
            group_count = 1

        box_labels, box_geometries, box_mask = self.stack_box_annotations(
            ground_truth,
            self.class_count
        )

        # Build the attention mask.
        N = len(boxes)
        G = group_count
        B = box_count_max
        device = boxes[0].device
        p = 0  # positive group index
        n = 1  # negative group index

        rep_box_labels = box_labels.tile((1, 2 * G))
        # Shape is (N, B * 2 * G)

        rep_box_geometries = box_labels.tile((1, 2 * G, 1))
        # Shape is (N, B * 2 * G, 4)

        rep_box_mask = box_mask.tile((1, 2 * G))
        # Shape is (N, B * 2 * G, 1)

        # Remember that the attention mask is the negation of the adjacency
        # graph, It tells which box pairs (i, j) are forbidden to interact with
        # each other.
        #
        # Each box from the same group can interact with each other
        negative = torch.zeros((N, 2 * B, 1), device=device)
        negative[:, B:] = 1
        negative = negative.tile((1, G, 1))
        positive = 1 - negative

        # shape is (N, 2 * G, 1)
        # [0, 0, 0, 0, 1, 1, 1, 1]
        # [0, 0, 1, 1, 1, 1, 1, 1]
        # [0, 0, 0, 0, 0, 0, 1, 1]
        positive = positive.squeeze(-1) * box_mask
        dn_positive_ixs = torch.nonzero(positive)

        return self.Output(box_labels, box_geometries, box_mask, {})


class BoxGeometryDenoiser(nn.Module):
    """The `BoxGeometryDenoiser` implements the auxiliary network that learns
    to denoised noised ground-truth object boxes as described in the paper
    [DN-DETR: Accelerate DETR Training by Introducing Query
    DeNoising](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_DN-DETR_Accelerate_DETR_Training_by_Introducing_Query_DeNoising_CVPR_2022_paper.pdf)

    This is an important module that makes the training of DETR-based
    architectures a lot faster as indicated in the title of the paper.
    """

    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(num_classes + 1,
                                      embedding_dim,
                                      padding_idx=num_classes)

        nn.init.normal_(self.embedding.weight[:-1])
