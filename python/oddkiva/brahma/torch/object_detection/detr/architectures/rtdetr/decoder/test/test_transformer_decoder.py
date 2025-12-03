from loguru import logger

import torch
import torch.nn.functional as F

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.decoder.query_decoder import (
        MultiScaleDeformableTransformerDecoderLayer
    )


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def relative_error(a: torch.Tensor, b: torch.Tensor):
    num = torch.dist(a, b)
    denom = torch.norm(a)
    return num / denom


def test_multiscale_deformable_attention_in_decoder_layer_0():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    decoder_data = data['intermediate']['decoder']
    # `gt` as in ground truth.
    gt = decoder_data['decoder.layers.0']
    # `msda` as in MultiscaleDeformableAttention.
    msda_gt = gt['cross_attn.deformable_attention_core_func_v2']

    memory, memory_spatial_hw_sizes = decoder_data['_get_encoder_input']
    query, query_geometry_logits, _, _ = decoder_data['_get_decoder_input']

    decoder = ckpt.load_transformer_decoder()

    layer = decoder.layers[0]
    assert type(layer) is MultiScaleDeformableTransformerDecoderLayer

    query_geometries = F.sigmoid(query_geometry_logits)
    query_geometries_true = gt['ref_points_detach']
    assert torch.dist(query_geometries, query_geometries_true) < 1e-12

    qgeom_embeds = \
        decoder.box_geometry_embedding_map(query_geometries)
    qgeom_embeds_true = gt['query_pos_embed']
    assert torch.dist(qgeom_embeds, qgeom_embeds_true) < 1e-12

    # We will check each step of this call.
    #
    # query_refined = layer.forward(
    #     query.detach(), query_geometries,
    #     memory, memory_spatial_hw_sizes,
    #     query_positional_embeds=query_geometry_embeds
    # )

    # Check the query and key matrices.
    q = k = layer.with_positional_embeds(query, qgeom_embeds)
    q_true = gt['q']
    k_true = gt['k']
    assert torch.dist(q, q_true) < 1e-12
    assert torch.dist(k, k_true) < 1e-12

    # Check the self-attention computation.
    self_attn_out, _ = layer.self_attention.forward(q, k, query, attn_mask=None)
    self_attn_out_true = gt['self_attn']
    assert torch.dist(self_attn_out, self_attn_out_true) < 1e-20

    #
    target2 = self_attn_out
    target = query

    # Check the 1st (dropout+add+layer-norm) computation.
    target = target + layer.dropout_1(target2)
    target = layer.layer_norm_1(target)
    target_true = gt['dropout+add+norm.1']
    assert torch.dist(target, target_true) < 1e-12

    # Check the cross-attention computation.
    query_x = layer.with_positional_embeds(target, qgeom_embeds)
    query_x_true = gt['cross_attn_query']
    assert torch.dist(query_x, query_x_true) < 1e-12

    x_attn_query_geometries_true = gt['cross_attn_query_geometries']
    assert torch.dist(
        query_geometries.flatten(),
        x_attn_query_geometries_true.flatten()) < 1e-12

    # Break down the cross-attention computations.
    #
    # 1.a. Project the memory. The projected memory is the value matrix V.
    V = layer.cross_attention.value_projector(memory)
    # 1.b. Reshape the value matrix.
    M = layer.cross_attention.attention_head_count
    d_v = layer.cross_attention.value_dim
    N, value_count, d_v_all = V.shape
    assert d_v_all == d_v * M
    V = V.reshape(N, value_count, M, d_v)
    V_true = gt['cross_attn.value_proj']
    assert torch.dist(V, V_true) < 1e-12
    # 2. Predict the sampling offsets.
    Δx_qmlk = layer.cross_attention.predict_positional_offsets(query_x)
    Δx_qmlk_true = gt['cross_attn.sampling_offsets']
    assert torch.dist(Δx_qmlk, Δx_qmlk_true) < 1e-12
    # 3. Predict the attention weights.
    w_qmlk = layer.cross_attention.predict_attention_weights(query_x)
    w_qmlk_true = gt['cross_attn.attention_weights']
    assert torch.dist(w_qmlk, w_qmlk_true) < 1e-12
    # 4. Calculate the sample positions.
    x_qmlk = layer.cross_attention.calculate_sample_positions(
        query_geometries, Δx_qmlk
    )
    x_qmlk_true = gt['cross_attn.sampling_locations']
    assert torch.dist(
        x_qmlk,
        x_qmlk_true
    ) < 1e-12

    V_mask = None
    if V_mask is None:
        V_masked = V
    else:
        V_masked = V_mask.to(V.dtype) * V

    V_pyramid_hw_sizes = memory_spatial_hw_sizes


    value_pyramid = layer.cross_attention.reconstruct_value_pyramid(
        V_masked,
        memory_spatial_hw_sizes
    )
    value_pyramid_true = msda_gt['value_list']
    assert len(value_pyramid) == len(value_pyramid_true)
    for (value, value_true) in zip(value_pyramid, value_pyramid_true):
        assert torch.dist(value, value_true.reshape(value.shape)) < 1e-12

    # 3. Split the list of value positions per image level.
    #
    #    `x_kv_per_level` is the list of value sample locations for each
    #    image level `l`.
    #
    #    x_per_level[l] has shape (N, top-K, M * K, 2).
    K = layer.cross_attention.kv_count_per_level
    _, top_K, _ = query.shape
    assert K == 4

    x_qmlk_rescaled = 2 * x_qmlk - 1
    x_qmlk_rescaled = x_qmlk_rescaled\
        .permute(0, 2, 1, 3, 4)\
        .flatten(start_dim=0, end_dim=1)
    x_qmlk_rescaled_true = msda_gt['sampling_grids']
    assert torch.dist(x_qmlk_rescaled, x_qmlk_rescaled_true) < 1e-12

    x_per_level = [
        x_qmlk_rescaled[:, :, K*i:K*(i+1), :]
        # The shape of the tensor (N * M, top-K, K, 2)

        for i in range(len(V_pyramid_hw_sizes))
    ]
    x_per_level_true = msda_gt['sampling_locations_list']
    for x, x_true in zip(x_per_level, x_per_level_true):
        assert torch.dist(x, x_true) < 1e-12

    values_per_level = []
    for x_l, query_map in zip(x_per_level, value_pyramid):
        # Collapse the pair of indices (attention head index, key index)
        # into a 1D index.
        values_l = F.grid_sample(query_map, x_l)
        # Shape is (N * M, d_v, top-K, K)

        # Make sure we permute the axes again to perform the attention
        # calculus.
        # 0. Shape is (N * M, d_v, top-K, K)
        # 1. Shape is (N, M, d_v, top-K, K)
        #              0  1    2      3  4
        # 2. Shape is (N, top-K, M, K, d_v)
        values_per_level.append(values_l)

    values_per_level_true = msda_gt['sampling_value_list']
    for (v, v_true) in zip(values_per_level, values_per_level_true):
        assert torch.dist(v, v_true) < 1e-12

    values_per_level = [
        v\
        .reshape(N, M, d_v, top_K, K)\
        .permute(0, 3, 1, 4, 2)
        for v in values_per_level
    ]

    value_qmlk = torch.cat(values_per_level, dim=3)
    value_qmlk_reweighted = value_qmlk * w_qmlk[..., None]
    value_qm = torch.sum(value_qmlk_reweighted, dim=3)
    value_q = value_qm.flatten(start_dim=2, end_dim=-1)
    value_q_backprojected = layer.cross_attention.backprojector(value_q)

    value_q_true = msda_gt['output']
    assert torch.dist(value_q, value_q_true) < 1e-5

    value_q_backprojected_true = gt['output_proj']
    assert torch.dist(value_q_backprojected, value_q_backprojected_true) < 5e-5


def test_transformer_decoder_layer_0_details():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    decoder_data = data['intermediate']['decoder']
    # `gt` as in ground truth.
    gt = decoder_data['decoder.layers.0']

    memory, memory_spatial_hw_sizes = decoder_data['_get_encoder_input']
    query, query_geometry_logits, _, _ = decoder_data['_get_decoder_input']

    decoder = ckpt.load_transformer_decoder()

    layer = decoder.layers[0]
    assert type(layer) is MultiScaleDeformableTransformerDecoderLayer

    query_geometries = F.sigmoid(query_geometry_logits)
    query_geometries_true = gt['ref_points_detach']
    assert torch.dist(query_geometries, query_geometries_true) < 1e-12

    qgeom_embeds = \
        decoder.box_geometry_embedding_map(query_geometries)
    qgeom_embeds_true = gt['query_pos_embed']
    assert torch.dist(qgeom_embeds, qgeom_embeds_true) < 1e-12

    # We will check each step of this call.
    #
    # query_refined = layer.forward(
    #     query.detach(), query_geometries,
    #     memory, memory_spatial_hw_sizes,
    #     query_positional_embeds=query_geometry_embeds
    # )

    # Check the query and key matrices.
    q = k = layer.with_positional_embeds(query, qgeom_embeds)
    q_true = gt['q']
    k_true = gt['k']
    assert torch.dist(q, q_true) < 1e-12
    assert torch.dist(k, k_true) < 1e-12

    # Check the self-attention computation.
    self_attn_out, _ = layer.self_attention.forward(q, k, query, attn_mask=None)
    self_attn_out_true = gt['self_attn']
    assert torch.dist(self_attn_out, self_attn_out_true) < 1e-20

    #
    target2 = self_attn_out
    target = query

    # Check the 1st (dropout+add+layer-norm) computation.
    target = target + layer.dropout_1(target2)
    target = layer.layer_norm_1(target)
    target_true = gt['dropout+add+norm.1']
    assert torch.dist(target, target_true) < 1e-12

    # Check the cross-attention computation.
    query_x = layer.with_positional_embeds(target, qgeom_embeds)
    query_x_true = gt['cross_attn_query']
    assert torch.dist(query_x, query_x_true) < 1e-12

    x_attn_query_geometries_true = gt['cross_attn_query_geometries']
    assert torch.dist(
        query_geometries.flatten(),
        x_attn_query_geometries_true.flatten()) < 1e-12

    memory_mask = None
    cross_attn_out = layer.cross_attention.forward(\
        layer.with_positional_embeds(target, qgeom_embeds),
        query_geometries,
        memory,
        memory_spatial_hw_sizes,
        memory_mask)
    cross_attn_out_true = gt['cross_attn']
    assert torch.dist(cross_attn_out, cross_attn_out_true) < 5e-5

    target2 = cross_attn_out

    # Check the 2nd (dropout+add+layer-norm) computation.
    target = target + layer.dropout_2(target2)
    target = layer.layer_norm_2(target)
    target_true = gt['dropout+add+norm.2']
    assert torch.dist(target, target_true) < 5e-5

    # FFN
    target2 = layer.feedforward.forward(target)
    target2_true = gt['ffn']
    assert torch.dist(target2, target2_true) < 1.5e-4

    target = target + layer.dropout_3(target2)
    target = layer.layer_norm_3(target)
    target_true = gt['dropout+add.norm.3']
    assert torch.dist(target, target_true) < 1.5e-4


def test_transformer_decoder_layer_0_api():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    decoder_data = data['intermediate']['decoder']
    # `gt` as in ground truth.
    gt = decoder_data['decoder.layers.0']

    memory, memory_spatial_hw_sizes = decoder_data['_get_encoder_input']
    query, query_geometry_logits, _, _ = decoder_data['_get_decoder_input']

    decoder = ckpt.load_transformer_decoder()

    layer = decoder.layers[0]
    assert type(layer) is MultiScaleDeformableTransformerDecoderLayer

    query_geometries = F.sigmoid(query_geometry_logits)
    query_geometries_true = gt['ref_points_detach']
    assert torch.dist(query_geometries, query_geometries_true) < 1e-12

    qgeom_embeds = \
        decoder.box_geometry_embedding_map(query_geometries)
    qgeom_embeds_true = gt['query_pos_embed']
    assert torch.dist(qgeom_embeds, qgeom_embeds_true) < 1e-12


    layer_out = layer.forward(query, query_geometries,
                              memory, memory_spatial_hw_sizes,
                              query_positional_embeds=qgeom_embeds)
    layer_out_true = gt['dropout+add.norm.3']
    assert torch.dist(layer_out, layer_out_true) < 1.5e-4


def test_transformer_decoder_details():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    memory, memory_spatial_hw_sizes = \
        data['intermediate']['decoder']['_get_encoder_input']
    init_ref_contents, init_ref_points_unact, _, _ = \
        data['intermediate']['decoder']['_get_decoder_input']

    decoder = ckpt.load_transformer_decoder()
    # box_geometries, box_class_logits = decoder.forward(
    #     init_ref_contents.detach(), init_ref_points_unact,
    #     memory, memory_spatial_hw_sizes,
    #     value_mask=None
    # )

    query = init_ref_contents
    query_geometry_logits = init_ref_points_unact
    value = memory
    value_spatial_sizes = memory_spatial_hw_sizes
    value_mask = None
    memory_mask = None
    assert query.requires_grad is False
    assert query_geometry_logits.requires_grad is False

    layers_gt = data['intermediate']['decoder']['decoder.layer-by-layer']

    # Get the actual query geometry by activating the logits with the
    # sigmoid function.

    # Initialize:
    # - the current query embedding
    # - the current query geometry logits (the object geometry before the
    #   sigmoid activation)
    # - the current query geometry (the object box geometry)
    query_curr = query
    query_geom_logits_curr = query_geometry_logits
    query_geom_curr = F.sigmoid(query_geom_logits_curr)

    query_next: torch.Tensor | None = None
    query_class_logits_next: torch.Tensor | None = None
    query_geom_logits_next: torch.Tensor | None = None
    query_geom_next: torch.Tensor | None = None

    query_geometries_denoised = []
    query_class_logits_denoised = []

    query_geometries_denoised_true: list[torch.Tensor] = layers_gt['dec_out_bboxes']
    query_class_logits_denoised_true: list[torch.Tensor] = layers_gt['dec_out_logits']

    for i, decoder_layer in enumerate(decoder.layers):
        logger.debug(f'[{i}] Checking decoder layer {i}')
        layer_gt_i = layers_gt[str(i)]

        assert type(decoder_layer) is \
            MultiScaleDeformableTransformerDecoderLayer

        # Calculate the corresponding embed vector of the box geometry
        query_geom_embed_curr = decoder.box_geometry_embedding_map\
            .forward(query_geom_curr)
        query_geom_embed_curr_true = layer_gt_i['query_pos_embed']
        assert torch.dist(query_geom_embed_curr,
                          query_geom_embed_curr_true) < 2.5e-3
        assert relative_error(query_geom_embed_curr,
                              query_geom_embed_curr_true) < 1e-5

        query_curr_true = layer_gt_i['input_ref_contents']
        assert torch.dist(query_curr, query_curr_true) < 2.5e-3
        assert relative_error(query_curr, query_curr_true) < 2e-5

        # Denoise the current query.
        query_next = decoder_layer.forward(
            query_curr, query_geom_curr,
            value, value_spatial_sizes,
            query_positional_embeds=query_geom_embed_curr,
            attn_mask=value_mask, memory_mask=memory_mask
        )
        query_next_true = layer_gt_i['output_ref_contents']
        assert torch.dist(query_next, query_next_true) < 2.5e-3
        assert relative_error(query_next, query_next_true) < 2e-5

        # Estimate the new object class logits (object class probabilities).
        query_class_logits_next = decoder.box_class_logit_heads[i](query_next)
        query_class_logits_next_true: torch.Tensor = query_class_logits_denoised_true[i]
        assert query_class_logits_next is not None
        assert torch.dist(query_class_logits_next,
                          query_class_logits_next_true) < 2.5e-3
        assert relative_error(query_class_logits_next,
                              query_class_logits_next_true) < 1e-5

        # Estimate the new object geometry (cx cy w h).
        Δ_query_geom_logits = decoder.box_geometry_logit_heads[i](query_next)
        query_geom_logits_next = \
            query_geom_logits_curr + Δ_query_geom_logits
        query_geom_next = F.sigmoid(query_geom_logits_next)
        query_geom_next_true: torch.Tensor = layer_gt_i['inter_ref_bbox']
        assert torch.dist(query_geom_next,
                          query_geom_next_true) < 2.5e-3
        assert relative_error(query_geom_next, query_geom_next_true) < 1e-5

        # Store the denoised results.
        query_geometries_denoised.append(query_geom_next)
        query_class_logits_denoised.append(query_class_logits_next)

        # Update for the next denoising iteration.
        query_curr = query_next
        query_geom_logits_curr = query_geom_logits_next
        # Make sure that we only optimize the residuals at the training
        # stage.
        query_geom_curr = query_geom_next.detach()


    for i, (qgeom, qgeom_true) in enumerate(zip(query_geometries_denoised,
                                                query_geometries_denoised_true)):
        logger.debug(f'[{i}] Checking iterative query geometry denoising {i}')
        assert torch.dist(qgeom, qgeom_true) < 2.5e-3
        assert relative_error(qgeom, qgeom_true) < 5e-6

    for i, (qc, qc_true) in enumerate(zip(query_class_logits_denoised,
                                          query_class_logits_denoised_true)):
        logger.debug(f'[{i}] Checking iterative query class denoising {i}')
        assert torch.dist(qc, qc_true) < 2.5e-3
        assert relative_error(qc, qc_true) < 5e-6


def test_transformer_decoder_api():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    memory, memory_spatial_hw_sizes = \
        data['intermediate']['decoder']['_get_encoder_input']
    init_ref_contents, init_ref_points_unact, _, _ = \
        data['intermediate']['decoder']['_get_decoder_input']

    decoder = ckpt.load_transformer_decoder()

    query = init_ref_contents
    query_geometry_logits = init_ref_points_unact
    value = memory
    value_spatial_sizes = memory_spatial_hw_sizes
    value_mask = None
    assert query.requires_grad is False
    assert query_geometry_logits.requires_grad is False

    layers_gt = data['intermediate']['decoder']['decoder.layer-by-layer']

    box_geometries, box_class_logits = decoder.forward(
        query, query_geometry_logits,
        value, value_spatial_sizes,
        value_mask=value_mask
    )
    box_geometries_true = torch.stack(layers_gt['dec_out_bboxes'])
    box_class_logits_true = torch.stack(layers_gt['dec_out_logits'])

    assert relative_error(box_geometries, box_geometries_true) < 5e-6
    assert relative_error(box_class_logits, box_class_logits_true) < 5e-6
