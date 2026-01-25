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

    # Collect the backbone learnable parameters.
    backbone_params = []
    for param_name, param in model.backbone.named_parameters():
        if 'batch_norm' in param_name:
            continue

        if not param.requires_grad:
            continue

        backbone_params.append((param_name, param))
        print(f'{param_name} shape: {param.shape}')

    backbone_lr_params = {
    }

    import ipdb; ipdb.set_trace()
