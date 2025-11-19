import math
from collections import OrderedDict

import torch
import torch.nn as nn

from oddkiva.brahma.torch.backbone.multi_layer_perceptron import (
    MultiLayerPerceptron
)
from oddkiva.brahma.torch.object_detection.common.anchors import (
    calculate_anchor_geometry_logits,
    enumerate_anchor_pyramid,
)


class AnchorGeometryLogitEnumerator(nn.Module):
    """
    This class enumerates all possible anchor boxes at every level of the image
    pyramid.

    The geometry of an anchor is encoded as a 4D vector, which is considered to
    be an *activated* value. The activation function we use is specifically the
    sigmoid function. It is an increasing function whose inverse function can
    be calculated analytically.

    The purpose of this class is to generate all the inverse sigmoid values and
    these are the so-called *logits*.
    """

    def __init__(self, normalized_base_size: float = 0.05, eps: float = 1e-2):
        """Constructs an `AnchorGeometryLogitEnumerator` object.

        Parameters:
            normalized_base_size:
                base box size relative to the image sizes
            eps:
                epsilon value used to filter out corner-case logit values
        """
        super().__init__()

        self.normalized_base_size = normalized_base_size
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pyramid_image_wh_sizes: list[tuple[int, int]],
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_pyramid = enumerate_anchor_pyramid(
            pyramid_image_wh_sizes,
            normalized_base_box_size=self.normalized_base_size,
            normalize_anchor_geometry=True,
            device=device
        )

        anchors = torch.cat(anchor_pyramid, dim=0)
        anchor_geometry_logits, anchor_mask = calculate_anchor_geometry_logits(
            anchors,
            eps=self.eps
        )
        return anchor_geometry_logits, anchor_mask


class AnchorGeomeryResidualHead(MultiLayerPerceptron):

    def __init__(self,
                 encoding_dim: int,
                 hidden_dim: int,
                 layer_count: int,
                 activation='relu'):
        super().__init__(
            encoding_dim, hidden_dim, 4,  # (Δcx, Δcy, Δw, Δh)
            layer_count, activation=activation
        )


class AnchorDecoder(nn.Module):
    """
    Feature vectors of the feature pyramid are then projected onto a feature
    space of the same dimention and finally concatenated into the so-called
    *memory* tensor.

    These feature vectors are the encoding of every possible anchors. These
    encoded anchors can be decoded so that we select the most promising ones.

    Namely, for each anchor, the decoder predicts its object class logits and
    its encoded box geometry.

    The object class logits will be used in another query selection module to
    rank and keep the best object queries from the memory tensor.
    """

    def __init__(self,
                 encoding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 geometry_head_layer_count: int = 3,
                 geometry_head_activation: str = 'relu',
                 normalized_base_size: float = 0.05,
                 logit_eps: float = 1e-2,
                 precalculate_anchor_geometry_logits: bool = True,
                 image_pyramid_wh_sizes: list[tuple[int, int]] | None = None,
                 device: torch.device | None = None,
                 initial_class_probability: float = 0.1):
        super().__init__()

        self.anchor_geometry_logit_enumerator = AnchorGeometryLogitEnumerator(
            normalized_base_size=normalized_base_size,
            eps=logit_eps
        )

        self._anchor_decoder_base = nn.Sequential(OrderedDict([
            ('projector', nn.Linear(hidden_dim, hidden_dim)),
            ('layer_norm', nn.LayerNorm(hidden_dim,))
        ]))

        self._class_logit_head = nn.Linear(hidden_dim, num_classes)

        self._geometry_residual_head = AnchorGeomeryResidualHead(
            encoding_dim, hidden_dim, geometry_head_layer_count,
            activation=geometry_head_activation
        )

        self._reinitialize_learning_parameters(initial_class_probability)

        self.precalculate_anchor_geometry_logits = \
            precalculate_anchor_geometry_logits
        if precalculate_anchor_geometry_logits:
            if image_pyramid_wh_sizes is None or device is None:
                raise ValueError()
            self.anchor_geometry_logits, self.anchor_mask = \
                self.anchor_geometry_logit_enumerator.forward(
                    image_pyramid_wh_sizes, device)
        else:
            self.anchor_geometry_logits = None
            self.anchor_mask = None

    def _initial_class_logit_value(self, prob: float) -> float:
        """ The interpretatiion is the probability that a
        """
        logit_value = -math.log((1 - prob) / prob)
        return logit_value

    def _reinitialize_learning_parameters(self,
                                          initial_class_probability: float):
        # Initialize the weight of the base network.
        nn.init.xavier_uniform_(self._anchor_decoder_base[0].weight)

        # Initialize the bias of the class logit head.
        nn.init.constant_(
            self._class_logit_head.bias,
            self._initial_class_logit_value(prob=initial_class_probability)
        )

        # Initialize the weight and bias of the last layer of the geometry
        # residual head.
        nn.init.constant_(self._geometry_residual_head.layers[-2].weight, 0)
        nn.init.constant_(self._geometry_residual_head.layers[-2].bias, 0)

    def forward(
        self,
        memory: torch.Tensor,
        feature_pyramid_wh_sizes: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.precalculate_anchor_geometry_logits:
            anchor_geometry_logits_refined, anchor_mask = (
                self.anchor_geometry_logits,
                self.anchor_mask
            )
            assert anchor_geometry_logits_refined is not None
            assert anchor_mask is not None
        else:
            anchor_geometry_logits_refined, anchor_mask = \
                self.anchor_geometry_logit_enumerator.forward(
                    feature_pyramid_wh_sizes,
                    memory.device
                )

        memory_filtered = anchor_mask.to(dtype=memory.dtype) * memory
        memory_filtered = self._anchor_decoder_base(memory_filtered)

        anchor_class_logits = self._class_logit_head(memory_filtered)
        anchor_geometry_residual_logits = \
            self._geometry_residual_head(memory_filtered)

        anchor_geometry_logits_refined = \
            anchor_geometry_logits_refined + anchor_geometry_residual_logits

        return (memory_filtered,
                anchor_class_logits,
                anchor_geometry_logits_refined)
