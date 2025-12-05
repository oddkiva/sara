import torch
import torch.nn as nn


class BoxObjectClassEmbedding(nn.Module):

    def __init__(self):
        super().__init__()


class BoxGeometryNoiser(nn.Module):

    def __init__(self,
                 box_count: int = 100,
                 box_noise_scale: float = 1.0):
        super().__init__()

        self.object_class_embedding_fn: nn.Module
        self.box_count = box_count
        self.box_noise_scale = box_noise_scale


    def forward(self, ground_truth_boxes: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        # TODO: understand first how COCO dataset is structurec.
        ...


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
