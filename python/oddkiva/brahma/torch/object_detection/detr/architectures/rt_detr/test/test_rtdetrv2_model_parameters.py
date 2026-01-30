import torch

from oddkiva.brahma.torch import DEFAULT_DEVICE
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.config import RTDETRConfig
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rt_detr.model import RTDETRv2
from oddkiva.brahma.torch.utils.freeze import freeze_batch_norm


def test_model_parameter_count():
    # THE DATA
    device = torch.device(DEFAULT_DEVICE)

    # THE MODEL
    config = RTDETRConfig()
    model = RTDETRv2(config).to(device)
    model = freeze_batch_norm(model)

    backbone = model.backbone
    bparams = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(bparams)
    # ME      : 23474016
    # ORIGINAL: 23445504


    params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )
    print(params)
    print(params)
    print(params)
    print(params)
    # It shows: 43863652
    # TODO: Find out why the number of trainable parameters: 42862860
    #
    # import ipdb; ipdb.set_trace()


def test_model_parameters():
    config = RTDETRConfig()
    model = RTDETRv2(config)

    assert [p for (_, p) in model.named_parameters()] == [*model.parameters()]

    # Collect the backbone learnable parameters.
    param_groups = model.group_learnable_parameters()

    assert len(param_groups[0]['params']) == len(model.backbone_learnable_params())
    assert len(param_groups[1]['params']) == len(model.encoder_learnable_params())
    assert len(param_groups[2]['params']) == len(model.query_selector_learnable_params())
    assert len(param_groups[3]['params']) == len(model.decoder_learnable_params())

    parameter_count = len([p for p in model.parameters() if p.requires_grad])
    remaining_parameter_count = \
        parameter_count - \
        sum(len(pg['params']) for pg in param_groups[:-1])
    assert len(param_groups[4]['params']) == remaining_parameter_count

    assert param_groups[0]['lr'] == 1e-5
    for i in range(1, 5):
        assert param_groups[i]['lr'] == 1e-4
