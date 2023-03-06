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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <drafts/NeuralNetworks/TensorRT/YoloImpl.hpp>


namespace DO::Sara::TensorRT {

  __global__ void yolo_kernel(const float* conv, float* yolo,     //
                              const int num_boxes_per_grid_cell,  //
                              const int grid_height,              //
                              const int grid_width,               //
                              const int num_classes,              //
                              const float scale_x_y)
  {
    // Bound checks.
    const auto box_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto num_boxes = num_boxes_per_grid_cell * grid_height * grid_width;
    if (box_idx >= num_boxes)
      return;

    // Let us explain the box features in YOLO-V4 tiny.
    static constexpr auto num_coordinates = 4;          // (x, y, w, h)
    const auto num_probabilities = 1 /* P[object] */ +  //
                                   num_classes /* P[class|object] */;
    const auto num_box_features = num_coordinates + num_probabilities;

    const auto hw = grid_height * grid_width;
    const auto fhw = num_box_features * hw;

    // YOLO calculates a 5D tensor [N, B, F, H, W]:
    // where:
    // - N = 1  is the batch size
    // - B = 3  is the number of box prediction per grid cell
    // - F = 85 is the number of box features
    //          (x, y, w, h, obj, class 0,... class 79)
    // - H = 13 is the grid height
    // - W = 13 is the grid width

    // Retrieve the 3D box index (b, i, j)
    const auto b = box_idx / hw;
    const auto ij = box_idx - b * hw;
#if DEBUG_YOLO_KERNEL
    const auto i = ij / w;
    const auto j = ij - i * w;
    printf("box_idx=%d, b=%d, i=%d, j=%d\n", box_idx, b, i, j);
#endif

    static constexpr auto logistic_fn = [](const float v) {
      return 1 / (1 + __expf(-v));
    };

    const auto flat_index = [fhw, hw, ij](const auto b, const auto f) {
      return fhw * b + hw * f + ij;
    };

    // The box coordinates.
    const auto x_idx = flat_index(b, 0);
    const auto y_idx = flat_index(b, 1);
    const auto& alpha = scale_x_y;
    const auto beta = -0.5f * (alpha - 1);
    yolo[x_idx] = alpha * logistic_fn(conv[x_idx]) + beta;
    yolo[y_idx] = alpha * logistic_fn(conv[y_idx]) + beta;

    // Recopy the values coordinates:
    const auto w_idx = flat_index(b, 2);
    const auto h_idx = flat_index(b, 3);
    yolo[w_idx] = conv[w_idx];
    yolo[h_idx] = conv[h_idx];

    // The probability that the box contains an object.
    const auto obj_idx = flat_index(b, 4);
    yolo[obj_idx] = logistic_fn(conv[obj_idx]);

    // The probability that the box is an object of class if it does contain
    // an object.
    for (auto class_id = 0; class_id < num_classes; ++class_id)
    {
      const auto prob_class_idx = flat_index(b, 5 + class_id);
      yolo[prob_class_idx] = logistic_fn(conv[prob_class_idx]);
    }
  }

  void yolo(const float* conv, float* yolo,     //
            const int num_boxes_per_grid_cell,  //
            const int grid_height,              //
            const int grid_width,               //
            const int num_classes,              //
            const float scale_x_y,              //
            cudaStream_t stream)
  {
    // CUDA kernels have very bad performance if there is branching in the
    // implementation.
    //
    // The scheduling strategy is to have each thread calculating one box.
    const auto total_num_boxes =
        num_boxes_per_grid_cell * grid_height * grid_width;
#if DEBUG_YOLO_KERNEL
    SARA_CHECK(num_boxes_per_grid_cell);
    SARA_CHECK(grid_height);
    SARA_CHECK(grid_width);
    SARA_CHECK(total_num_boxes);
#endif

    // By design CUDA can have at most 1024 threads per block, so let us use this
    // limit.
    static constexpr auto max_threads_per_block = 1024;
    const auto num_blocks = total_num_boxes % 1024 == 0
                                ? total_num_boxes / max_threads_per_block
                                : total_num_boxes / max_threads_per_block + 1;
#if DEBUG_YOLO_KERNEL
    SARA_CHECK(num_blocks);
#endif

    yolo_kernel<<<num_blocks, max_threads_per_block, 0, stream>>>(
        conv, yolo, num_boxes_per_grid_cell, grid_height, grid_width,
        num_classes, scale_x_y);
  }
}  // namespace DO::Sara::TensorRT
