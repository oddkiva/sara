import torch
import torch.nn as nn


class ObjectNoiser(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, ground_truth_boxes: torch.Tensor) -> torch.Tensor:
        ...


class ObjectQueryDenoiser(nn.Module):
    """The `ObjectQueryDenoiser` implements the auxiliary network that learns
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

    def 
