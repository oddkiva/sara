from pathlib import Path

from loguru import logger

import torch
import torch.nn as nn
import torchvision.ops as ops
from torch.serialization import MAP_LOCATION

from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm
from oddkiva.brahma.torch.backbone.resnet.rtdetrv2_variant import (
    UnbiasedConvBNA,
    ResNet50RTDETRV2Variant
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.feature_pyramid_projection import FeaturePyramidProjection
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.aifi import AIFI
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.ccff import (
        CCFF,
        DownsampleConvolution,
        FusionBlock,
        LateralConvolution,
        TopDownFusionNet,
        BottomUpFusionNet
    )
from oddkiva.brahma.torch.transformers.encoder import (
    TransformerEncoderLayer
)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.encoder.hybrid_encoder import HybridEncoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.anchor_decoder import AnchorDecoder
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_decoder import (BoxGeometryEmbeddingMap,
                                         BoxGeometryLogitHead,
                                         BoxObjectClassLogitHead)
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.multiscale_deformable_attention import (
        MultiscaleDeformableAttention
    )
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_decoder import (
        MultiScaleDeformableTransformerDecoderLayer,
        MultiScaleDeformableTransformerDecoder
    )
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.rtdetrv2_decoder import RTDETRv2Decoder


class RTDETRV2Checkpoint:

    resnet_arch_levels = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    batch_norm_param_names = ['weight', 'bias', 'running_mean', 'running_var']

    def __init__(self, ckpt_fp: Path, map_location: MAP_LOCATION = None):
        self.ckpt = torch.load(ckpt_fp, map_location=map_location)

    @property
    def model_weights(self):
        return self.ckpt['ema']['module']

    # -------------------------------------------------------------------------
    # BACKBONE
    # -------------------------------------------------------------------------
    @property
    def backbone_keys(self):
        return [k for k in self.model_weights.keys() if 'backbone' in k]

    @property
    def backbone_weights(self):
        return {k: self.model_weights[k] for k in self.backbone_weights}

    def conv1_key(self, subblock_idx: int):
        return f'backbone.conv1.conv1_{subblock_idx}'

    def conv1_conv_key(self, subblock_idx: int):
        return f"{self.conv1_key(subblock_idx)}.conv"

    def conv1_bn_key(self, subblock_idx: int):
        return f"{self.conv1_key(subblock_idx)}.norm"

    def conv1_conv_weight(self, subblock_idx: int) -> torch.Tensor:
        key = f"{self.conv1_conv_key(subblock_idx)}.weight"
        return self.model_weights[key]

    def conv1_bn_weights(
        self, subblock_idx: int
    ) -> dict[str, torch.Tensor]:
        keys = {
            param: f"{self.conv1_bn_key(subblock_idx)}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def bottleneck_key(self, block_idx: int, subblock_idx: int) -> str:
        return f'backbone.res_layers.{block_idx}.blocks.{subblock_idx}'

    def bottleneck_branch_key(self, block_idx: int, subblock_idx: int,
                              branch: str) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.branch2{branch}'

    def bottleneck_branch_conv_key(
        self, block_idx: int, subblock_idx: int, branch: str
    ) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.branch2{branch}.conv'

    def bottleneck_branch_conv_weight(
        self, block: int, subblock: int, branch: str
    ) -> torch.Tensor:
        parent_key = self.bottleneck_branch_conv_key(block, subblock, branch)
        key = f"{parent_key}.weight"
        return self.model_weights[key]

    def bottleneck_branch_bn_key(
        self, block_idx: int, subblock_idx: int, branch: str
    ) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.branch2{branch}.norm'

    def bottleneck_branch_bn_weights(
        self, block: int, subblock: int, branch: str
    ) -> dict[str, torch.Tensor]:
        parent_key = self.bottleneck_branch_bn_key(block, subblock, branch)
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def bottleneck_short_key(self, block_idx: int, subblock_idx: int) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        return f'{bottleneck_key}.short'

    def bottleneck_short_conv_key(self, block_idx: int, subblock_idx: int) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        if block_idx == 0:
            return f'{bottleneck_key}.short.conv'
        else:
            return f'{bottleneck_key}.short.conv.conv'

    def bottleneck_short_bn_key(self, block_idx: int, subblock_idx: int) -> str:
        bottleneck_key = self.bottleneck_key(block_idx, subblock_idx)
        if block_idx == 0:
            return f'{bottleneck_key}.short.norm'
        else:
            return f'{bottleneck_key}.short.conv.norm'

    def bottleneck_short_conv_weight(
        self, block: int, subblock: int
    ) -> torch.Tensor:
        parent_key = self.bottleneck_short_conv_key(block, subblock)
        key = f"{parent_key}.weight"
        return self.model_weights[key]

    def bottleneck_short_bn_weights(
        self, block: int, subblock: int
    ) -> dict[str, torch.Tensor]:
        parent_key = self.bottleneck_short_bn_key(block, subblock)
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    # -------------------------------------------------------------------------
    # ENCODER
    # -------------------------------------------------------------------------
    @property
    def encoder_keys(self):
        return [k for k in self.model_weights.keys() if 'encoder' in k]

    @property
    def encoder_weights(self):
        return {k: self.model_weights[k] for k in self.encoder_weights}

    @property
    def encoder_input_proj_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys if 'encoder.input_proj' in k}

    def encoder_input_proj_weight_key(self, i: int):
        return f'encoder.input_proj.{i}'

    def encoder_input_proj_conv_weight(self, i: int):
        key = f'encoder.input_proj.{i}.conv.weight'
        return self.model_weights[key]

    def encoder_input_proj_bn_weights(self, i: int):
        parent_key = f'encoder.input_proj.{i}.norm'
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    @property
    def encoder_aifi_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys if 'encoder.encoder' in k}

    @property
    def aifi_layer_suffixes(self):
        return ['self_attn.in_proj_weight',
                'self_attn.in_proj_bias',
                'self_attn.out_proj.weight',
                'self_attn.out_proj.bias',
                'linear1.weight',
                'linear1.bias',
                'linear2.weight',
                'linear2.bias',
                'norm1.weight',
                'norm1.bias',
                'norm2.weight',
                'norm2.bias']

    def aifi_key(self, i: int = 0):
        return f'encoder.encoder.{i}'

    def aifi_transformer_encoder_layer_key(self, layer_idx: int, i: int = 0):
        return f'{self.aifi_key(i)}.layers.{layer_idx}'

    def aifi_encoder_layer_weights(
        self, layer_idx: int, i: int = 0
    ) -> dict[str, torch.Tensor]:
        weight_keys = {
            suffix: '.'.join([
                self.aifi_transformer_encoder_layer_key(layer_idx, i),
                suffix
            ])
            for suffix in self.aifi_layer_suffixes
        }
        weights = {
            suffix: self.model_weights[weight_keys[suffix]]
            for suffix in self.aifi_layer_suffixes
        }
        return weights

    @property
    def encoder_lateral_conv_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys if 'encoder.lateral_convs' in k}

    def encoder_lateral_conv_weight_key(self, i: int) -> str:
        return f'encoder.lateral_convs.{i}.conv.weight'

    def encoder_lateral_conv_weight(self, i: int):
        return self.model_weights[self.encoder_lateral_conv_weight_key(i)]

    def encoder_lateral_bn_weights(self, i: int):
        parent_key = f'encoder.lateral_convs.{i}.norm'
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    @property
    def encoder_fpn_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys if 'encoder.fpn_blocks' in k}

    def encoder_fpn_conv_weight(self, fusion_idx: int, conv_idx: int):
        conv_base_key = f'encoder.fpn_blocks.{fusion_idx}'
        conv_weight_key = f'{conv_base_key}.conv{conv_idx}.conv.weight'
        return self.model_weights[conv_weight_key]

    def encoder_fpn_bn_weights(self, fusion_idx: int, conv_idx: int):
        conv_base_key = f'encoder.fpn_blocks.{fusion_idx}'
        bn_base_key = f'{conv_base_key}.conv{conv_idx}.norm'
        keys = {
            param: f"{bn_base_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def encoder_fpn_fusion_conv_weight(self,
                                       repvgg_stack_idx: int,
                                       repvgg_block_idx: int,
                                       conv_idx: int):
        conv_key = '.'.join([
            f'encoder.fpn_blocks.{repvgg_stack_idx}',
            f'bottlenecks.{repvgg_block_idx}',
            f'conv{conv_idx}.conv.weight'
        ])
        return self.model_weights[conv_key]

    def encoder_fpn_fusion_bn_weights(self,
                                      repvgg_stack_idx: int,
                                      repvgg_block_idx: int,
                                      conv_idx: int):
        parent_key = '.'.join([
            f'encoder.fpn_blocks.{repvgg_stack_idx}',
            f'bottlenecks.{repvgg_block_idx}',
            f'conv{conv_idx}.norm'
        ])
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    @property
    def encoder_downsample_conv_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys
                if 'encoder.downsample_convs' in k}

    def encoder_downsample_conv_weight_key(self, i: int) -> str:
        return f'encoder.downsample_convs.{i}.conv.weight'

    def encoder_downsample_conv_weight(self, i: int):
        return self.model_weights[self.encoder_downsample_conv_weight_key(i)]

    def encoder_downsample_bn_weights(self, i: int):
        parent_key = f'encoder.downsample_convs.{i}.norm'
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def encoder_pan_conv_weight(self, fusion_idx: int, conv_idx: int):
        conv_base_key = f'encoder.pan_blocks.{fusion_idx}'
        conv_weight_key = f'{conv_base_key}.conv{conv_idx}.conv.weight'
        return self.model_weights[conv_weight_key]

    def encoder_pan_bn_weights(self, fusion_idx: int, conv_idx: int):
        conv_base_key = f'encoder.pan_blocks.{fusion_idx}'
        bn_base_key = f'{conv_base_key}.conv{conv_idx}.norm'
        keys = {
            param: f"{bn_base_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def encoder_pan_fusion_conv_weight(self,
                                       repvgg_stack_idx: int,
                                       repvgg_block_idx: int,
                                       conv_idx: int):
        conv_key = '.'.join([
            f'encoder.pan_blocks.{repvgg_stack_idx}',
            f'bottlenecks.{repvgg_block_idx}',
            f'conv{conv_idx}.conv.weight'
        ])
        return self.model_weights[conv_key]

    def encoder_pan_fusion_bn_weights(self,
                                      repvgg_stack_idx: int,
                                      repvgg_block_idx: int,
                                      conv_idx: int):
        parent_key = '.'.join([
            f'encoder.pan_blocks.{repvgg_stack_idx}',
            f'bottlenecks.{repvgg_block_idx}',
            f'conv{conv_idx}.norm'
        ])
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    # -------------------------------------------------------------------------
    # DECODER
    # -------------------------------------------------------------------------
    @property
    def decoder_keys(self):
        return [k for k in self.model_weights.keys() if 'decoder' in k]

    @property
    def decoder_weights(self):
        return {k: self.model_weights[k] for k in self.decoder_weights}

    @property
    def decoder_input_proj_weights(self):
        return {k: self.model_weights[k]
                for k in self.decoder_keys if 'decoder.input_proj' in k}

    def decoder_input_proj_weight_key(self, i: int):
        return f'decoder.input_proj.{i}'

    def decoder_input_proj_conv_weight(self, i: int):
        key = f'decoder.input_proj.{i}.conv.weight'
        return self.model_weights[key]

    def decoder_input_proj_bn_weights(self, i: int):
        parent_key = f'decoder.input_proj.{i}.norm'
        keys = {
            param: f"{parent_key}.{param}"
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }
        return {
            param: self.model_weights[keys[param]]
            for param in RTDETRV2Checkpoint.batch_norm_param_names
        }

    def decoder_enc_output_proj_weight(self):
        key = f'decoder.enc_output.proj.weight'
        return self.model_weights[key]

    def decoder_enc_output_proj_bias(self):
        key = f'decoder.enc_output.proj.bias'
        return self.model_weights[key]

    def decoder_enc_output_norm_weight(self):
        key = f'decoder.enc_output.norm.weight'
        return self.model_weights[key]

    def decoder_enc_output_norm_bias(self):
        key = f'decoder.enc_output.norm.bias'
        return self.model_weights[key]

    def decoder_enc_score_head_weight(self):
        key = f'decoder.enc_score_head.weight'
        return self.model_weights[key]

    def decoder_enc_score_head_bias(self):
        key = f'decoder.enc_score_head.bias'
        return self.model_weights[key]

    def decoder_enc_bbox_head_weight(self, i: int):
        key = f'decoder.enc_bbox_head.layers.{i}.weight'
        return self.model_weights[key]

    def decoder_enc_bbox_head_bias(self, i: int):
        key = f'decoder.enc_bbox_head.layers.{i}.bias'
        return self.model_weights[key]

    def decoder_query_pos_head_weight(self, i: int):
        key = f'decoder.query_pos_head.layers.{i}.weight'
        return self.model_weights[key]

    def decoder_query_pos_head_bias(self, i: int):
        key = f'decoder.query_pos_head.layers.{i}.bias'
        return self.model_weights[key]

    def decoder_score_head_weight(self, iteration: int):
        key = f'decoder.dec_score_head.{iteration}.weight'
        return self.model_weights[key]

    def decoder_score_head_bias(self, iteration: int):
        key = f'decoder.dec_score_head.{iteration}.bias'
        return self.model_weights[key]

    def decoder_bbox_head_weight(self, iteration: int, layer_idx: int):
        key = f'decoder.dec_bbox_head.{iteration}.layers.{layer_idx}.weight'
        return self.model_weights[key]

    def decoder_bbox_head_bias(self, iteration: int, layer_idx: int):
        key = f'decoder.dec_bbox_head.{iteration}.layers.{layer_idx}.bias'
        return self.model_weights[key]

    def decoder_decoder_base_key(self) -> str:
        return 'decoder.decoder'

    def decoder_decoder_layer_base_key(self, iteration: int) -> str:
        return f'decoder.decoder.layers.{iteration}'

    @property
    def self_attn_layer_suffixes(self):
        return [
            'self_attn.in_proj_weight',
            'self_attn.in_proj_bias',
            'self_attn.out_proj.weight',
            'self_attn.out_proj.bias'
        ]

    def decoder_layer_self_attn_weights(
        self, iteration: int
    ) -> dict[str, torch.Tensor]:
        weight_keys = {
            suffix: '.'.join([
                self.decoder_decoder_layer_base_key(iteration),
                suffix
            ])
            for suffix in self.self_attn_layer_suffixes
        }
        weights = {
            suffix: self.model_weights[weight_keys[suffix]]
            for suffix in self.self_attn_layer_suffixes
        }
        return weights

    # -------------------------------------------------------------------------
    # WEIGHT COPY UTILITIES
    # -------------------------------------------------------------------------
    def _copy_conv_bna_weights(self, my_block: UnbiasedConvBNA,
                               conv_weight: torch.Tensor,
                               bn_weights: dict[str, torch.Tensor]) -> None:
        my_conv = my_block.layers[0]
        assert type(my_conv) is nn.Conv2d
        my_bn = my_block.layers[1]
        assert isinstance(my_bn, nn.BatchNorm2d | ops.FrozenBatchNorm2d)

        assert my_conv.weight.shape == conv_weight.shape
        assert my_bn.weight.shape == bn_weights['weight'].shape
        assert my_bn.bias.shape == bn_weights['bias'].shape
        assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
        assert my_bn.running_var.shape == bn_weights['running_var'].shape

        my_conv.weight.data.copy_(conv_weight)
        my_bn.weight.data.copy_(bn_weights['weight'])
        my_bn.bias.data.copy_(bn_weights['bias'])
        my_bn.running_mean.data.copy_(bn_weights['running_mean'])
        my_bn.running_var.data.copy_(bn_weights['running_var'])

        assert torch.equal(my_conv.weight, conv_weight)
        assert torch.equal(my_bn.weight, bn_weights['weight'])
        assert torch.equal(my_bn.bias, bn_weights['bias'])
        assert torch.equal(my_bn.running_mean, bn_weights['running_mean'])
        assert torch.equal(my_bn.running_var, bn_weights['running_var'])

    def _copy_self_attn_weights(
        self,
        self_attn: nn.MultiheadAttention,
        weights: dict[str, torch.Tensor]
    ) -> None:
        assert self_attn.in_proj_weight.shape == \
            weights['self_attn.in_proj_weight'].shape
        assert self_attn.in_proj_bias.shape == \
            weights['self_attn.in_proj_bias'].shape
        assert self_attn.out_proj.weight.shape == \
            weights['self_attn.out_proj.weight'].shape
        assert self_attn.out_proj.bias.shape == \
            weights['self_attn.out_proj.bias'].shape

        self_attn.in_proj_weight.data.copy_(weights['self_attn.in_proj_weight'])
        self_attn.in_proj_bias.data.copy_(weights['self_attn.in_proj_bias'])
        self_attn.out_proj.weight.data.copy_(weights['self_attn.out_proj.weight'])
        self_attn.out_proj.bias.data.copy_(weights['self_attn.out_proj.bias'])

        assert torch.equal(self_attn.in_proj_weight,
                           weights['self_attn.in_proj_weight'])
        assert torch.equal(self_attn.in_proj_bias,
                           weights['self_attn.in_proj_bias'])
        assert torch.equal(self_attn.out_proj.weight,
                           weights['self_attn.out_proj.weight'])
        assert torch.equal(self_attn.out_proj.bias,
                           weights['self_attn.out_proj.bias'])

    def _copy_weight_and_bias(self,
                              module: nn.Linear | nn.LayerNorm,
                              weight: torch.Tensor,
                              bias: torch.Tensor) -> None:
        assert module.weight.shape == weight.shape
        assert module.bias.shape == bias.shape
        module.weight.data.copy_(weight)
        module.bias.data.copy_(bias)
        assert torch.equal(module.weight, weight)
        assert torch.equal(module.bias, bias)

    # -------------------------------------------------------------------------
    # BACKBONE LOAD UTILITIES
    # -------------------------------------------------------------------------
    def _load_backbone_conv_1(self, model):
        logger.info('Loading RT-DETR v2 backbone.conv_1')
        for i in range(3):
            my_conv_bna = model.blocks[0][i]

            conv_weight = self.conv1_conv_weight(i + 1)
            bn_weights = self.conv1_bn_weights(i + 1)
            self._copy_conv_bna_weights(my_conv_bna, conv_weight, bn_weights)

    def _load_backbone_res_layers(self, model):
        logger.info('Loading RT-DETR v2 backbone.conv_1')

        # Convenient variable names.
        indexing = {0: 'a', 1: 'b', 2: 'c'}
        resnet50_arch_levels = self.resnet_arch_levels[50]

        for bottleneck_stack_idx, bottleneck_count in enumerate(resnet50_arch_levels):

            for i in range(bottleneck_count):
                logger.debug(f"block_idx = {bottleneck_stack_idx}  i = {i}")
                my_bottleneck_stack = model.blocks[bottleneck_stack_idx + 1]
                my_bottleneck_block = my_bottleneck_stack[i]

                # Loop through ['branch2a', 'branch2b', 'branch2c']
                for j in range(3):
                    my_convbna = my_bottleneck_block.convs[j]
                    assert type(my_convbna) is UnbiasedConvBNA

                    my_conv = my_bottleneck_block.convs[j].layers[0]
                    my_bn = my_bottleneck_block.convs[j].layers[1]

                    letter = indexing[j]
                    conv_weight = self.bottleneck_branch_conv_weight(bottleneck_stack_idx, i, letter)
                    bn_weights = self.bottleneck_branch_bn_weights(bottleneck_stack_idx, i, letter)

                    assert my_conv.weight.shape == conv_weight.shape
                    assert my_bn.weight.shape == bn_weights['weight'].shape
                    assert my_bn.bias.shape == bn_weights['bias'].shape
                    assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
                    assert my_bn.running_var.shape == bn_weights['running_var'].shape

                    logger.info(
                        ''.join(
                            f'Loading conv+bn weights for '
                            f'(s:{bottleneck_stack_idx}, b:{i}, l:{letter})'
                        )
                    )
                    self._copy_conv_bna_weights(my_convbna,
                                                conv_weight, bn_weights)

                if i == 0:
                    # Only the first bottleneck in each stack has a shortcut
                    # connection.
                    my_shortcut = my_bottleneck_block.shortcut
                    if type(my_shortcut) is UnbiasedConvBNA:
                        my_convbna = my_shortcut
                    elif type(my_shortcut) is torch.nn.Sequential:
                        my_convbna = my_shortcut[1]
                        assert type(my_convbna) is UnbiasedConvBNA
                    else:
                        raise TypeError("This should not happen")

                    my_conv = my_convbna.layers[0]
                    my_bn = my_convbna.layers[1]

                    conv_weight = self.bottleneck_short_conv_weight(bottleneck_stack_idx, i)
                    bn_weights = self.bottleneck_short_bn_weights(bottleneck_stack_idx, i)

                    assert my_conv.weight.shape == conv_weight.shape
                    assert my_bn.weight.shape == bn_weights['weight'].shape
                    assert my_bn.bias.shape == bn_weights['bias'].shape
                    assert my_bn.running_mean.shape == bn_weights['running_mean'].shape
                    assert my_bn.running_var.shape == bn_weights['running_var'].shape

                    self._copy_conv_bna_weights(my_convbna,
                                                conv_weight, bn_weights)

                else:
                    # There are no shortcut connection in this bottleneck
                    # block.
                    my_shortcut = my_bottleneck_block.shortcut
                    assert type(my_shortcut) is nn.Identity

    def load_backbone(self) -> ResNet50RTDETRV2Variant:
        model = ResNet50RTDETRV2Variant()
        model = freeze_batch_norm(model)
        assert type(model) is ResNet50RTDETRV2Variant

        self._load_backbone_conv_1(model)
        self._load_backbone_res_layers(model)

        return model

    # -------------------------------------------------------------------------
    # ENCODER LOAD UTILITIES
    # -------------------------------------------------------------------------
    def load_encoder_input_proj(self) -> FeaturePyramidProjection:
        # Just hardcode the variables to simplify.
        fp_proj = FeaturePyramidProjection(
            [512, 1024, 2048],
            256
        )
        fp_proj = freeze_batch_norm(fp_proj)
        assert type(fp_proj) is FeaturePyramidProjection

        # Copy the model weights.
        for i in range(3):
            my_convbna = fp_proj.projections[i]
            conv_weight = self.encoder_input_proj_conv_weight(i)
            bn_weights = self.encoder_input_proj_bn_weights(i)
            assert type(my_convbna) is UnbiasedConvBNA
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        return fp_proj

    def load_encoder_aifi(self) -> AIFI:
        hidden_dim = 256
        attn_head_count = 8
        feedforward_dim = 1024
        num_layers = 1
        dropout = 0.
        normalize_before = False

        aifi = AIFI(hidden_dim,
                    attn_head_count,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    num_layers=num_layers,
                    norm=None)

        for layer_idx in range(num_layers):
            weights = self.aifi_encoder_layer_weights(layer_idx, i=0)
            layer = aifi.transformer_encoder.layers[layer_idx]
            assert type(layer) is TransformerEncoderLayer

            # Self-attention weights.
            self._copy_self_attn_weights(layer.self_attention, weights)
            # Feed-forward weights.
            ffn_linear_1 = layer.feedforward[0]
            ffn_linear_2 = layer.feedforward[3]
            self._copy_weight_and_bias(ffn_linear_1,
                                       weights['linear1.weight'],
                                       weights['linear1.bias'])
            self._copy_weight_and_bias(ffn_linear_2,
                                       weights['linear2.weight'],
                                       weights['linear2.bias'])
            # Copy the layer norms 1 and 2.
            self._copy_weight_and_bias(layer.layer_norm_1,
                                       weights['norm1.weight'],
                                       weights['norm1.bias'])
            self._copy_weight_and_bias(layer.layer_norm_2,
                                       weights['norm2.weight'],
                                       weights['norm2.bias'])

        return aifi

    def load_encoder_lateral_convs(self) -> nn.ModuleList:
        lateral_convs = nn.ModuleList([
            freeze_batch_norm(LateralConvolution(256, 256))
            for _ in range(2)
        ])

        for idx in range(2):
            my_convbna = lateral_convs[idx]
            assert type(my_convbna) is LateralConvolution
            conv_weight = self.encoder_lateral_conv_weight(idx)
            bn_weights = self.encoder_lateral_bn_weights(idx)
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        return lateral_convs

    def load_encoder_top_down_fusion_blocks(self) -> nn.ModuleList:
        fusions = nn.ModuleList([
            freeze_batch_norm(FusionBlock(
                512, 256,
                hidden_dim_expansion_factor=1.0,
                repvgg_layer_count=3,
                activation='silu'
            ))
            for _ in range(2)
        ])

        for fusion_idx in range(2):
            fusion = fusions[fusion_idx]
            convs = [fusion.conv1, fusion.conv2]
            repvgg_stack = fusion.repvgg_stack

            # Copy the weights of conv1 and conv2
            for conv_idx in range(2):
                conv = convs[conv_idx]
                conv_weight = self.encoder_fpn_conv_weight(fusion_idx, conv_idx + 1)
                bn_weights = self.encoder_fpn_bn_weights(fusion_idx, conv_idx + 1)
                self._copy_conv_bna_weights(conv, conv_weight, bn_weights)

            # Copy the weights of RepVggStack
            for rep_block_idx in range(len(repvgg_stack.layers)):
                repvgg = repvgg_stack.layers[rep_block_idx]
                repvgg_conv3 = repvgg.layers[0]
                repvgg_conv1 = repvgg.layers[1]

                repvgg_conv3_weight = self.encoder_fpn_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 1
                )
                repvgg_bn3_weights = self.encoder_fpn_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 1
                )
                self._copy_conv_bna_weights(repvgg_conv3,
                                            repvgg_conv3_weight,
                                            repvgg_bn3_weights)

                repvgg_conv1_weight = self.encoder_fpn_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 2
                )
                repvgg_bn1_weights = self.encoder_fpn_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 2
                )
                self._copy_conv_bna_weights(repvgg_conv1,
                                            repvgg_conv1_weight,
                                            repvgg_bn1_weights)

        return fusions

    def load_encoder_top_down_fusion_network(self) -> TopDownFusionNet:
        top_down_fusion = freeze_batch_norm(TopDownFusionNet(
            512, 256, 2,
            hidden_dim_expansion_factor=1.0,
            repvgg_stack_depth=3,
            activation='silu'
        ))
        assert type(top_down_fusion) is TopDownFusionNet

        for idx in range(2):
            my_convbna = top_down_fusion.lateral_convs[idx]
            assert type(my_convbna) is LateralConvolution
            conv_weight = self.encoder_lateral_conv_weight(idx)
            bn_weights = self.encoder_lateral_bn_weights(idx)
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        for fusion_idx in range(2):
            logger.debug('fusion_idx = {fusion_idx}')
            fusion = top_down_fusion.fusion_blocks[fusion_idx]
            convs = [fusion.conv1, fusion.conv2]
            repvgg_stack = fusion.repvgg_stack

            # Copy the weights of conv1 and conv2
            for conv_idx in range(2):
                conv = convs[conv_idx]
                conv_weight = self.encoder_fpn_conv_weight(fusion_idx, conv_idx + 1)
                bn_weights = self.encoder_fpn_bn_weights(fusion_idx, conv_idx + 1)
                self._copy_conv_bna_weights(conv, conv_weight, bn_weights)

            # Copy the weights of RepVggStack
            for rep_block_idx in range(len(repvgg_stack.layers)):
                repvgg = repvgg_stack.layers[rep_block_idx]
                repvgg_conv3 = repvgg.layers[0]
                repvgg_conv1 = repvgg.layers[1]

                repvgg_conv3_weight = self.encoder_fpn_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 1
                )
                repvgg_bn3_weights = self.encoder_fpn_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 1
                )
                self._copy_conv_bna_weights(repvgg_conv3,
                                            repvgg_conv3_weight,
                                            repvgg_bn3_weights)

                repvgg_conv1_weight = self.encoder_fpn_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 2
                )
                repvgg_bn1_weights = self.encoder_fpn_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 2
                )
                self._copy_conv_bna_weights(repvgg_conv1,
                                            repvgg_conv1_weight,
                                            repvgg_bn1_weights)

        return top_down_fusion

    def load_encoder_downsample_convs(self) -> nn.ModuleList:
        downsample_convs = nn.ModuleList([
            freeze_batch_norm(  # IMPORTANT!
                DownsampleConvolution(256, 256)
            )
            for _ in range(2)
        ])

        for idx in range(2):
            my_convbna = downsample_convs[idx]
            assert type(my_convbna) is DownsampleConvolution
            conv_weight = self.encoder_downsample_conv_weight(idx)
            bn_weights = self.encoder_downsample_bn_weights(idx)
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        return downsample_convs

    def load_encoder_bottom_up_fusion_blocks(self) -> nn.ModuleList:
        fusions = nn.ModuleList([
            freeze_batch_norm(  # IMPORTANT!
                FusionBlock(
                    512, 256,
                    hidden_dim_expansion_factor=1.0,
                    repvgg_layer_count=3,
                    activation='silu'
                )
            )
            for _ in range(2)
        ])

        for fusion_idx in range(2):
            fusion = fusions[fusion_idx]
            convs = [fusion.conv1, fusion.conv2]
            repvgg_stack = fusion.repvgg_stack

            # Copy the weights of conv1 and conv2
            for conv_idx in range(2):
                conv = convs[conv_idx]
                conv_weight = self.encoder_pan_conv_weight(fusion_idx, conv_idx + 1)
                bn_weights = self.encoder_pan_bn_weights(fusion_idx, conv_idx + 1)
                self._copy_conv_bna_weights(conv, conv_weight, bn_weights)

            # Copy the weights of RepVggStack
            for rep_block_idx in range(len(repvgg_stack.layers)):
                repvgg = repvgg_stack.layers[rep_block_idx]
                repvgg_conv3 = repvgg.layers[0]
                repvgg_conv1 = repvgg.layers[1]

                repvgg_conv3_weight = self.encoder_pan_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 1
                )
                repvgg_bn3_weights = self.encoder_pan_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 1
                )
                self._copy_conv_bna_weights(repvgg_conv3,
                                            repvgg_conv3_weight,
                                            repvgg_bn3_weights)

                repvgg_conv1_weight = self.encoder_pan_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 2
                )
                repvgg_bn1_weights = self.encoder_pan_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 2
                )
                self._copy_conv_bna_weights(repvgg_conv1,
                                            repvgg_conv1_weight,
                                            repvgg_bn1_weights)

        return fusions

    def load_encoder_bottom_up_fusion_network(self) -> BottomUpFusionNet:
        bottom_up_fusion = freeze_batch_norm(BottomUpFusionNet(
            512, 256, 2,
            hidden_dim_expansion_factor=1.0,
            repvgg_stack_depth=3,
            activation='silu'
        ))
        assert type(bottom_up_fusion) is BottomUpFusionNet

        for idx in range(2):
            my_convbna = bottom_up_fusion.downsample_convs[idx]
            assert type(my_convbna) is DownsampleConvolution
            conv_weight = self.encoder_downsample_conv_weight(idx)
            bn_weights = self.encoder_downsample_bn_weights(idx)
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        for fusion_idx in range(2):
            logger.debug(f'fusion_idx = {fusion_idx}')
            fusion = bottom_up_fusion.fusion_blocks[fusion_idx]
            convs = [fusion.conv1, fusion.conv2]
            repvgg_stack = fusion.repvgg_stack

            # Copy the weights of conv1 and conv2
            for conv_idx in range(2):
                conv = convs[conv_idx]
                conv_weight = self.encoder_pan_conv_weight(fusion_idx, conv_idx + 1)
                bn_weights = self.encoder_pan_bn_weights(fusion_idx, conv_idx + 1)
                self._copy_conv_bna_weights(conv, conv_weight, bn_weights)

            # Copy the weights of RepVggStack
            for rep_block_idx in range(len(repvgg_stack.layers)):
                repvgg = repvgg_stack.layers[rep_block_idx]
                repvgg_conv3 = repvgg.layers[0]
                repvgg_conv1 = repvgg.layers[1]

                repvgg_conv3_weight = self.encoder_pan_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 1
                )
                repvgg_bn3_weights = self.encoder_pan_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 1
                )
                self._copy_conv_bna_weights(repvgg_conv3,
                                            repvgg_conv3_weight,
                                            repvgg_bn3_weights)

                repvgg_conv1_weight = self.encoder_pan_fusion_conv_weight(
                    fusion_idx,rep_block_idx, 2
                )
                repvgg_bn1_weights = self.encoder_pan_fusion_bn_weights(
                    fusion_idx,rep_block_idx, 2
                )
                self._copy_conv_bna_weights(repvgg_conv1,
                                            repvgg_conv1_weight,
                                            repvgg_bn1_weights)

        return bottom_up_fusion

    def load_encoder_ccff(self) -> CCFF:
        ccff = freeze_batch_norm(CCFF(
            2,
            256,
            hidden_dim_expansion_factor=1.0,
            repvgg_stack_depth=3,
            activation='silu'
        ))
        ccff.fuse_topdown = self.load_encoder_top_down_fusion_network()
        ccff.refine_bottomup = self.load_encoder_bottom_up_fusion_network()
        assert type(ccff) is CCFF
        return ccff

    def load_encoder(self) -> HybridEncoder:
        input_feature_dims = [512, 1024, 2048]
        hidden_dim = 256
        attn_head_count = 8
        attn_feedforward_dim = 1024
        attn_dropout = 0.
        attn_num_layers = 1

        encoder = freeze_batch_norm(HybridEncoder(
            input_feature_dims,
            attn_head_count,
            hidden_dim,
            attn_feedforward_dim=attn_feedforward_dim,
            attn_dropout=attn_dropout,
            attn_num_layers=attn_num_layers
        ))
        encoder.backbone_feature_proj = self.load_encoder_input_proj()
        encoder.aifi = self.load_encoder_aifi()
        encoder.ccff = self.load_encoder_ccff()
        assert type(encoder) is HybridEncoder
        return encoder

    # -------------------------------------------------------------------------
    # DECODER LOAD UTILITIES
    # -------------------------------------------------------------------------
    def load_decoder_input_proj(self) -> FeaturePyramidProjection:
        # Just hardcode the variables to simplify.
        fp_proj = FeaturePyramidProjection(
            [256] * 3,
            256
        )
        fp_proj = freeze_batch_norm(fp_proj)
        assert type(fp_proj) is FeaturePyramidProjection

        # Copy the model weights.
        for i in range(3):
            my_convbna = fp_proj.projections[i]
            conv_weight = self.decoder_input_proj_conv_weight(i)
            bn_weights = self.decoder_input_proj_bn_weights(i)
            assert type(my_convbna) is UnbiasedConvBNA
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        return fp_proj

    def load_decoder_anchor_decoder(self) -> AnchorDecoder:
        encoding_dim = 256
        hidden_dim = 256
        num_classes = 80

        anchor_decoder = AnchorDecoder(
            encoding_dim, hidden_dim, num_classes,
            geometry_head_layer_count=3,
            geometry_head_activation='relu',
            normalized_base_size=0.05,
            logit_eps=1e-2,
            precalculate_anchor_geometry_logits=True,
            image_pyramid_wh_sizes=[(80, 80), (40, 40), (20, 20)],
            device = torch.device('cpu'),
            initial_class_probability=0.1
        )
        anchor_decoder = freeze_batch_norm(anchor_decoder)

        self._copy_weight_and_bias(
            anchor_decoder.decoder_base.projector,
            self.decoder_enc_output_proj_weight(),
            self.decoder_enc_output_proj_bias()
        )
        self._copy_weight_and_bias(
            anchor_decoder.decoder_base.layer_norm,
            self.decoder_enc_output_norm_weight(),
            self.decoder_enc_output_norm_bias()
        )

        self._copy_weight_and_bias(
            anchor_decoder.class_logit_head,
            self.decoder_enc_score_head_weight(),
            self.decoder_enc_score_head_bias()
        )

        for i in range(anchor_decoder.geometry_residual_head.layer_count):
            linear = anchor_decoder.geometry_residual_head.layers[i]
            assert type(linear) is nn.Linear
            self._copy_weight_and_bias(
                linear,
                self.decoder_enc_bbox_head_weight(i),
                self.decoder_enc_bbox_head_bias(i)
            )

        assert type(anchor_decoder) is AnchorDecoder
        return anchor_decoder

    def load_box_geometry_embedding_map(self) -> BoxGeometryEmbeddingMap:
        embed_dim = 256
        embed_fn = BoxGeometryEmbeddingMap(embed_dim)
        for i, layer in enumerate(embed_fn.layers):
            assert type(layer) is nn.Linear
            self._copy_weight_and_bias(
                layer,
                self.decoder_query_pos_head_weight(i),
                self.decoder_query_pos_head_bias(i)
            )

        return embed_fn

    def load_box_geometry_logit_heads(self) -> list[BoxGeometryLogitHead]:
        embed_dim = 256
        iterations = 6
        num_layers = 3

        heads = []
        for i in range(iterations):
            head = BoxGeometryLogitHead(embed_dim, num_layers)

            for layer_idx, layer in enumerate(head.layers):
                assert type(layer) is nn.Linear
                self._copy_weight_and_bias(
                    layer,
                    self.decoder_bbox_head_weight(i, layer_idx),
                    self.decoder_bbox_head_bias(i, layer_idx)
                )

            heads.append(head)

        return heads

    def load_box_class_logit_heads(self) -> list[BoxObjectClassLogitHead]:
        embed_dim = 256
        num_classes = 80
        iterations = 6

        heads = []
        for i in range(iterations):
            head = BoxObjectClassLogitHead(embed_dim, num_classes)
            self._copy_weight_and_bias(
                head,
                self.decoder_score_head_weight(i),
                self.decoder_score_head_bias(i)
            )
            heads.append(head)

        return heads

    def load_transformer_decoder_layer_cross_attention(
        self, iteration: int
    ) -> MultiscaleDeformableAttention:
        embed_dim = 256
        num_heads = 8
        value_dim = 32

        msda = MultiscaleDeformableAttention(
            embed_dim, num_heads, value_dim,
            pyramid_level_count=3,
            kv_count_per_level=4
        )

        parent_key = f'decoder.decoder.layers.{iteration}.cross_attn'

        key = f'{parent_key}.sampling_offsets'
        self._copy_weight_and_bias(
            msda.sampling_offset_predictors,
            self.model_weights[f'{key}.weight'],
            self.model_weights[f'{key}.bias']
        )

        key = f'{parent_key}.attention_weights'
        self._copy_weight_and_bias(
            msda.attn_weight_predictors,
            self.model_weights[f'{key}.weight'],
            self.model_weights[f'{key}.bias']
        )

        key = f'{parent_key}.value_proj'
        self._copy_weight_and_bias(
            msda.value_projector,
            self.model_weights[f'{key}.weight'],
            self.model_weights[f'{key}.bias']
        )

        key = f'{parent_key}.output_proj'
        self._copy_weight_and_bias(
            msda.backprojector,
            self.model_weights[f'{key}.weight'],
            self.model_weights[f'{key}.bias']
        )

        return msda

    def load_transformer_decoder_layer_self_attention(
        self, iteration: int
    ) -> nn.MultiheadAttention:
        embed_dim = 256
        num_heads = 8

        self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.,
                                          batch_first=True)
        self._copy_self_attn_weights(
            self_attn,
            self.decoder_layer_self_attn_weights(iteration)
        )
        return self_attn

    def _copy_decoder_layer_feedforward_weights(
        self, ffn: nn.Sequential, iteration: int
    ) -> None:
        # Load the remaining weights
        assert type(ffn.linear1) is nn.Linear
        assert type(ffn.linear2) is nn.Linear

        parent_key = f'decoder.decoder.layers.{iteration}'
        w1 = self.model_weights[f'{parent_key}.linear1.weight']
        b1 = self.model_weights[f'{parent_key}.linear1.bias']
        self._copy_weight_and_bias(ffn.linear1, w1, b1)

        w2 = self.model_weights[f'{parent_key}.linear2.weight']
        b2 = self.model_weights[f'{parent_key}.linear2.bias']
        self._copy_weight_and_bias(ffn.linear2, w2, b2)

    def _copy_decoder_layer_norm_weights(
        self,
        dlayer: MultiScaleDeformableTransformerDecoderLayer,
        iteration: int
    ) -> None:
        parent_key = f'decoder.decoder.layers.{iteration}'

        w1 = self.model_weights[f'{parent_key}.norm1.weight']
        b1 = self.model_weights[f'{parent_key}.norm1.bias']
        self._copy_weight_and_bias(dlayer.layer_norm_1, w1, b1)

        w2 = self.model_weights[f'{parent_key}.norm2.weight']
        b2 = self.model_weights[f'{parent_key}.norm2.bias']
        self._copy_weight_and_bias(dlayer.layer_norm_2, w2, b2)

        w3 = self.model_weights[f'{parent_key}.norm3.weight']
        b3 = self.model_weights[f'{parent_key}.norm3.bias']
        self._copy_weight_and_bias(dlayer.layer_norm_3, w3, b3)

    def load_transformer_decoder_layer(
        self, iteration: int
    ) -> MultiScaleDeformableTransformerDecoderLayer:
        hidden_dim = 256
        image_level_count = 3
        attn_head_count = 8
        feedforward_dim = 1024
        dropout = 0.
        normalize_before = False

        dlayer = MultiScaleDeformableTransformerDecoderLayer(
            hidden_dim,
            attn_head_count,
            image_level_count,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            normalize_before=normalize_before,
        )

        dlayer.self_attention = \
            self.load_transformer_decoder_layer_self_attention(
                iteration
            )

        dlayer.cross_attention = \
            self.load_transformer_decoder_layer_cross_attention(
                iteration
            )

        self._copy_decoder_layer_feedforward_weights(
            dlayer.feedforward,
            iteration
        )
        self._copy_decoder_layer_norm_weights(dlayer, iteration)

        return dlayer

    def load_transformer_decoder(
        self
    ) -> MultiScaleDeformableTransformerDecoder:
        hidden_dim = 256
        kv_count_per_level = [4, 4, 4]
        num_classes = 80
        attn_head_count = 8
        attn_num_layers = 6
        attn_dropout = 0.0
        normalize_before = False

        decoder = MultiScaleDeformableTransformerDecoder(
            hidden_dim,
            kv_count_per_level,
            num_classes=num_classes,
            attn_head_count=attn_head_count,
            attn_num_layers=attn_num_layers,
            attn_dropout=attn_dropout,
            normalize_before=normalize_before
        )

        for i in range(attn_num_layers):
            decoder.layers[i] = self.load_transformer_decoder_layer(i)

        decoder.box_geometry_embedding_map = \
            self.load_box_geometry_embedding_map()

        decoder.box_class_logit_heads = nn.ModuleList(
            self.load_box_class_logit_heads()
        )

        decoder.box_geometry_logit_heads = nn.ModuleList(
            self.load_box_geometry_logit_heads()
        )

        return decoder

    def load_decoder(self) -> RTDETRv2Decoder:
        num_classes = 80
        encoding_dim = 256
        hidden_dim = 256
        pyramid_level_count = 3

        decoder = RTDETRv2Decoder(
            num_classes,
            encoding_dim,
            hidden_dim,
            pyramid_level_count,
            precalculate_anchor_geometry_logits=False
        )

        decoder.feature_projectors = self.load_decoder_input_proj()

        return decoder
