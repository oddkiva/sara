from pathlib import Path

from loguru import logger

from oddkiva.brahma.torch.object_detection.detr.architectures.rtdetr.encoder.aifi import AIFI
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
    rtdetr.encoder.backbone_feature_pyramid_projection import (
        BackboneFeaturePyramidProjection
    )


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
    def encoder_layer_key_suffixes(self):
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
        return f'{self.aifi_key(0)}.layers.{layer_idx}'

    def aifi_encoder_layer_weights(
        self, layer_idx: int, i: int = 0
    ) -> dict[str, torch.Tensor]:
        weight_keys = {
            suffix: '.'.join([
                self.aifi_transformer_encoder_layer_key(layer_idx, i),
                suffix
            ])
            for suffix in self.encoder_layer_key_suffixes
        }
        weights = {
            suffix: self.model_weights[weight_keys[suffix]]
            for suffix in self.encoder_layer_key_suffixes
        }
        return weights

    @property
    def encoder_lateral_conv_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys if 'encoder.lateral_convs' in k}

    @property
    def encoder_fpn_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys if 'encoder.fpn_blocks' in k}

    @property
    def encoder_downsample_conv_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_keys
                if 'encoder.downsample_convs' in k}

    @property
    def encoder_pan_weights(self):
        return {k: self.model_weights[k]
                for k in self.encoder_weights if 'encoder.pan' in k}

    # -------------------------------------------------------------------------
    # DECODER
    # -------------------------------------------------------------------------
    @property
    def decoder_keys(self):
        return [k for k in self.model_weights.keys() if 'decoder' in k]

    @property
    def decoder_weights(self):
        return {k: self.model_weights[k] for k in self.decoder_weights}

    # -------------------------------------------------------------------------
    # WEIGHT COPY UTILITIES
    # -------------------------------------------------------------------------
    def _copy_conv_bna_weights(self, my_block: UnbiasedConvBNA,
                               conv_weight: torch.Tensor,
                               bn_weights: dict[str, torch.Tensor]) -> None:
        my_conv: nn.Conv2d = my_block.layers[0]
        my_bn: nn.BatchNorm2d | ops.FrozenBatchNorm2d = my_block.layers[1]

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
                print(f"block_idx = {bottleneck_stack_idx}  i = {i}")
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
                        TypeError("This should not happen")

                    my_conv = my_convbna.layers[0]
                    my_bn = my_convbna.layers[1]

                    # TODO: copy the weight and bias.

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

    def load_backbone(self) -> nn.Module:
        model = ResNet50RTDETRV2Variant()
        model = freeze_batch_norm(model)

        self._load_backbone_conv_1(model)
        self._load_backbone_res_layers(model)

        return model

    def load_encoder_input_proj(self) -> nn.Module:
        # Just hardcode the variables to simplify.
        fp_proj = BackboneFeaturePyramidProjection(
            [512, 1024, 2048],
            256
        )
        fp_proj = freeze_batch_norm(fp_proj)

        # Copy the model weights.
        for i in range(3):
            my_convbna = fp_proj.projections[i]
            conv_weight = self.encoder_input_proj_conv_weight(i)
            bn_weights = self.encoder_input_proj_bn_weights(i)
            assert type(my_convbna) is UnbiasedConvBNA
            self._copy_conv_bna_weights(my_convbna, conv_weight, bn_weights)

        return fp_proj

    def load_encoder_aifi(self):
        hidden_dim = 256
        attn_head_count = 8
        feedforward_dim = 1024
        num_layers = 1
        dropout = 0.
        normalize_before = False
        norm = nn.GELU()

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
                                       weights['norm1.bias']),
            self._copy_weight_and_bias(layer.layer_norm_2,
                                       weights['norm2.weight'],
                                       weights['norm2.bias']),

        return aifi
