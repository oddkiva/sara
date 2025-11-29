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


def test_transformer_decoder_layer_0_and_multiscale_deformable_attention():
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
        query_geometry_logits, Δx_qmlk
    )
    x_qmlk_true = gt['cross_attn.sampling_locations']
    assert torch.dist(
        x_qmlk,
        x_qmlk_true.permute(0, 2, 1, 3, 4).flatten(0, 1)
    ) < 1e-12

    V_mask = None
    if V_mask is None:
        V_masked = V
    else:
        V_masked = V_mask.to(V.dtype) * V

    V_pyramid_hw_sizes = memory_spatial_hw_sizes


    value_pyramid = layer.cross_attention.reconstruct_value_pyramid(
        V,
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




    # # We have for each query `q` the `M*L*K` 32D value vectors, each one of
    # # them being indexed by:
    # # - the attention head `m`,
    # # - the image level `l`,
    # # - the key index `k`.
    # V_qmlk = layer.cross_attention.sample_values(
    #     V_masked, V_pyramid_hw_sizes,
    #     x_qlk
    # )

    # # For each query `q`, we have `M` composite value vectors, each one of them
    # # being indexed by the each attention head `m`
    # V_mq = torch.sum(w_qlk[..., None] * V_qmlk, dim=3)
    # assert V_mq.shape == (1, 300, 8, 32)

    # # The value is the refined object query vector.
    # V_mq = V_mq.flatten(start_dim=2)
    # V_q = layer.cross_attention.final_projections(V_mq)
    # assert V_q.shape == (1, 300, 256)

    # # cross_attn_out = layer.cross_attention.forward(\
    # #     layer.with_positional_embeds(target, qgeom_embeds),
    # #     query_geometries,
    # #     memory,
    # #     memory_spatial_hw_sizes,
    # #     memory_mask)
    # cross_attn_out = V_q
    # cross_attn_out_true = gt['cross_attn']
    # assert torch.dist(cross_attn_out.flatten(),
    #                   cross_attn_out_true.flatten()) < 1e-12

    # # Check the 2nd (dropout+add+layer-norm) computation.
    # target = target + layer.dropout_2(target2)
    # target = layer.layer_norm_2(target)
    # target_true = gt['dropout+add+norm.2']
    # assert torch.norm(target - target_true) < 1e-12

    # # FFN
    # target2 = layer.feedforward.forward(target)
    # target2_true = gt['ffn']
    # assert torch.norm(target2 - target2_true) < 1e-12

    # target = target + layer.dropout_3(target2)
    # target = layer.layer_norm_3(target)
    # target_true = gt['dropout+add.norm.3']
    # assert torch.norm(target - target_true) < 1e-12


# def test_transformer_decoder():
#     ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
#     data = torch.load(DATA_FILEPATH, torch.device('cpu'))
#
#     memory, memory_spatial_hw_sizes = \
#         data['intermediate']['decoder']['_get_encoder_input']
#     init_ref_contents, init_ref_points_unact, _, _ = \
#         data['intermediate']['decoder']['_get_decoder_input']
#
#     box_geometries_true, box_class_logits_true = \
#         data['intermediate']['decoder']['decoder']
#
#     decoder = ckpt.load_transformer_decoder()
#     box_geometries, box_class_logits = decoder.forward(
#         init_ref_contents.detach(), init_ref_points_unact,
#         memory, memory_spatial_hw_sizes,
#         value_mask=None
#     )
#
#     assert torch.norm(box_geometries - box_geometries_true) < 1e-12
#     assert torch.norm(box_class_logits - box_class_logits_true) < 1e-12
