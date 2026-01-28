from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2


# def test_model_parameter_count():
#     # THE DATA
#     device = torch.device(DEFAULT_DEVICE)
#
#     # THE MODEL
#     config = RTDETRConfig()
#     model = RTDETRv2(config).to(device)
#     model = freeze_batch_norm(model)
#
#     backbone = model.backbone
#     bparams = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
#     print(bparams)
#     # ME      : 23474016
#     # ORIGINAL: 23445504
#
#
#     #' params = sum(
#     #'     param.numel()
#     #'     for param in model.parameters()
#     #'     if param.requires_grad
#     #' )
#     #' print(params)
#     #' print(params)
#     #' print(params)
#     #' print(params)
#     #' # It shows: 43863652
#
#     #' # TODO: Find out Number of trainable parameters: 42862860


def test_model_parameters():
    config = RTDETRConfig()
    model = RTDETRv2(config)

    assert [p for (_, p) in model.named_parameters()] == [*model.parameters()]

    # Collect the backbone learnable parameters.
    backbone_params = model.backbone_learnable_params()
    query_selector_params = model.query_selector_learnable_params()
    encoder_params = model.encoder_learnable_params()
    decoder_params = model.decoder_learnable_params()

    selected_params = {
        **backbone_params,
        **query_selector_params,
        **encoder_params,
        **decoder_params,
    }

    remaining_params = {}
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param_name in selected_params:
            continue
        remaining_params[param_name] = param
        print(f'[RT-DETR v2] {param_name}: {param.shape}')
