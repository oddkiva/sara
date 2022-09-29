// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <driver_types.h>


namespace DO::Sara::TensorRT {

  void yolo(const float* conv, float* yolo,  //
            const int num_boxes_per_grid_cell,  //
            const int grid_height,              //
            const int grid_width,               //
            const int num_classes,              //
            const float scale_x_y,              //
            cudaStream_t stream);

}  // namespace DO::Sara::TensorRT
