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

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara::Darknet {

  struct YoloBox
  {
    Eigen::Vector4f box;
    float objectness_prob;
    Eigen::VectorXf class_probs;
  };

  //! @brief Post-processing functions.
  //! @{
  auto get_yolo_boxes(const TensorView_<float, 3>& output,
                      const std::vector<int>& box_sizes_prior,
                      const std::vector<int>& masks,
                      const Eigen::Vector2i& network_input_sizes,
                      const Eigen::Vector2i& original_sizes,
                      float objectness_threshold) -> std::vector<YoloBox>;

  // Simple greedy NMS based on area IoU criterion.
  auto nms(const std::vector<YoloBox>& detections, float iou_threshold = 0.4f)
      -> std::vector<YoloBox>;
  //! @}

}  // namespace DO::Sara::Darknet
