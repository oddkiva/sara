import torch

from oddkiva import DATA_DIR_PATH
from oddkiva.brahma.torch.object_detection.detr.architectures.\
    rtdetr.checkpoint import RTDETRV2Checkpoint


CKPT_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.pth')
DATA_FILEPATH = (DATA_DIR_PATH / 'model-weights' / 'rtdetrv2' /
                 'rtdetrv2_r50vd_6x_coco_ema.data.pt')


def test_decoder_computations():
    ckpt = RTDETRV2Checkpoint(CKPT_FILEPATH, torch.device('cpu'))
    data = torch.load(DATA_FILEPATH, torch.device('cpu'))

    decoder = ckpt.load_decoder()

    fpyramid = data['intermediate']['encoder']['out']

    fpyramid_projected = decoder.feature_projectors(fpyramid)
    fpyramid_projected_gt = data['intermediate']['decoder']['input_proj']
    with torch.no_grad():
        for fpyr, fpyr_true in zip(fpyramid_projected, fpyramid_projected_gt):
            assert torch.dist(fpyr, fpyr_true) < 1e-12
            assert torch.dist(fpyr, fpyr_true, p=torch.inf) < 1e-12

    memory = decoder._transform_feature_pyramid_into_memory(fpyramid_projected)
    memory_image_sizes = [
        fmap.shape[2:][::-1]
        for fmap in fpyramid_projected
    ]

    memory_true, memory_image_sizes_true = \
        data['intermediate']['decoder']['_get_encoder_input']

    with torch.no_grad():
        assert torch.dist(memory, memory_true) < 1e-12
    for shape, shape_true in zip(memory_image_sizes,
                                 memory_image_sizes_true):
        assert shape == torch.Size(shape_true)
