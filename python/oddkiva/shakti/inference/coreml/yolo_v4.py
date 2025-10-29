# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

from pathlib import Path

import torch

import coremltools as ct

import oddkiva.shakti.inference.darknet as darknet


def convert_yolo_v4_to_coreml(yolo_net: darknet.Network,
                              in_tensor: torch.Tensor,
                              path: Path):
    with torch.inference_mode():
        traced_model = torch.jit.trace(yolo_net, in_tensor)
        outs = traced_model(in_tensor)

        ct_ins = [ct.ImageType(name="image",
                               shape=in_tensor.shape,
                               scale=1 / 255)]
        ct_outs = [ct.TensorType(name=f'yolo_{i}') for i, _ in enumerate(outs)]

        model = ct.convert(traced_model, inputs=ct_ins, outputs=ct_outs,
                           debug=True)

        model.input_description["image"] = "Input RGB image"
        for i in range(len(outs)):
            model.output_description[f"yolo_{i}"] =\
                f"Box predictions at scale {i}"

        model.save(path)
