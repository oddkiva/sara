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
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <boost/algorithm/string.hpp>


namespace DO::Sara::Darknet {

  struct Layer
  {
    virtual ~Layer() = default;

    virtual auto parse_line(const std::string&) -> void = 0;

    virtual auto to_output_stream(std::ostream& os) const -> void = 0;

    virtual auto forward(const TensorView_<float, 4>&)
        -> const TensorView_<float, 4>&
    {
      throw std::runtime_error{"Unimplemented!"};
      return output;
    }

    friend auto operator<<(std::ostream& os, const Layer& l) -> std::ostream&
    {
      l.to_output_stream(os);
      return os;
    }

    bool debug = false;
    std::string type;
    Eigen::Vector4i input_sizes = Eigen::Vector4i::Constant(-1);
    Eigen::Vector4i output_sizes = Eigen::Vector4i::Constant(-1);

    Tensor_<float, 4> output;
  };

  struct Input : Layer
  {
    auto update_output_sizes(bool inference = true) -> void
    {
      if (inference)
        batch() = 1;
      output_sizes(1) = 3;
      output.resize(output_sizes);
    }

    auto parse_line(const std::string& line) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;

    auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override
    {
      output = x;
      return output;
    }

    auto width() noexcept -> int&
    {
      return output_sizes(3);
    };

    auto height() noexcept -> int&
    {
      return output_sizes(2);
    }

    auto batch() noexcept -> int&
    {
      return output_sizes(0);
    }

    auto width() const noexcept -> const int&
    {
      return output_sizes(3);
    };

    auto height() const noexcept -> const int&
    {
      return output_sizes(2);
    }

    auto batch() const noexcept -> const int&
    {
      return output_sizes(0);
    }
  };

  struct BatchNormalization : Layer
  {
    struct Weights
    {
      Eigen::VectorXf scales;
      Eigen::VectorXf rolling_mean;
      Eigen::VectorXf rolling_variance;
    } weights;

    auto resize(const Eigen::Vector4i& sizes) -> void
    {
      const auto& num_filters = sizes(1);
      input_sizes = sizes;
      output_sizes = sizes;
      weights.scales.resize(num_filters);
      weights.rolling_mean.resize(num_filters);
      weights.rolling_variance.resize(num_filters);

      output.resize(sizes);
    }

    auto parse_line(const std::string&) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;

    auto load_weights(FILE* fp) -> void;
  };

  struct Convolution : Layer
  {
    bool batch_normalize = false;

    int filters;
    int size;
    int stride;
    int pad;
    std::string activation;

    struct Weights
    {
      Tensor_<float, 4> w;
      Eigen::VectorXf b;
    } weights;

    std::unique_ptr<BatchNormalization> bn_layer;

    auto update_output_sizes() -> void;

    auto parse_line(const std::string& line) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;

    auto load_weights(FILE* fp, bool inference = true) -> void;

    auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override;
  };

  //! @brief Concatenates the output of the different layers into a single
  //! output.
  struct Route : Layer
  {
    std::vector<std::int32_t> layers;
    int groups = 1;
    int group_id = -1;

    auto update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes)
        -> void;

    auto parse_line(const std::string& line) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;
  };


  struct Shortcut : Layer
  {
    int from;
    std::string activation;

    auto update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes)
        -> void;

    auto parse_line(const std::string&) -> void override;

    auto to_output_stream(std::ostream&) const -> void override;

    auto forward(const TensorView_<float, 4>& fx, const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>&;
  };

  struct MaxPool : Layer
  {
    int size = 2;
    int stride = 2;

    auto update_output_sizes() -> void;

    auto parse_line(const std::string& line) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;

    auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override;
  };

  struct Upsample : Layer
  {
    int stride = 2;

    auto update_output_sizes() -> void;

    auto parse_line(const std::string& line) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;

    auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override;
  };

  struct Yolo : Layer
  {
    //! @brief  The list of anchor box sizes (w[0], h[0]), ..., (w[N-1], h[N-1])
    std::vector<std::int32_t> anchors;
    //! @brief The list of anchor box indices to consider.
    std::vector<std::int32_t> mask;
    //! @brief The number of object classes.
    std::int32_t classes;

    std::int32_t num;
    float jitter;
    float scale_x_y;
    float cls_normalizer;
    float iou_normalizer;
    std::string iou_loss;
    float ignore_thresh;
    float truth_thresh;
    int random;
    float resize;
    std::string nms_kind;
    float beta_nms;

    auto update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes)
        -> void;

    auto parse_line(const std::string& line) -> void override;

    auto to_output_stream(std::ostream& os) const -> void override;

    auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override;
  };

}  // namespace DO::Sara::Darknet
