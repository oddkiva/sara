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

#include <drafts/NeuralNetworks/Darknet/Network.hpp>
#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <NvInfer.h>

#include <stdexcept>


namespace DO::Sara::TensorRT {

  struct YoloV4Converter
  {
    using TrtNet = nvinfer1::INetworkDefinition;
    using HostNet = std::vector<std::unique_ptr<Darknet::Layer>>;

    TrtNet* tnet;
    const HostNet& hnet;

    YoloV4Converter(TrtNet* tnet, const HostNet& hnet)
      : tnet{tnet}
      , hnet{hnet}
    {
    }

    auto make_input_tensor(const int c, const int h, const int w) const
        -> nvinfer1::ITensor*;

    auto make_input_rgb_tensor(const int w, const int h) const
        -> nvinfer1::ITensor*;

    //! @brief zero-padding convolution.
    auto conv2d(nvinfer1::ITensor* x,  //
                const TensorView_<float, 4>& w, const Eigen::VectorXf& b,
                const int stride, const std::string& activation_layer,
                const std::optional<std::string>& name = std::nullopt) const
        -> nvinfer1::ITensor*;

    auto add_conv2d_layer(const int layer_idx,
                          std::vector<nvinfer1::ITensor*>& fmaps) const -> void;

    auto add_slice_layer(const int layer_idx,
                         std::vector<nvinfer1::ITensor*>& fmaps) const -> void;

    auto add_concat_layer(const int layer_idx,
                          std::vector<nvinfer1::ITensor*>& fmaps) const -> void;

    auto add_shortcut_layer(const int layer_idx,
                            std::vector<nvinfer1::ITensor*>& fmaps) const
        -> void;

    auto add_maxpool_layer(const int layer_idx,
                           std::vector<nvinfer1::ITensor*>& fmaps) const
        -> void;

    auto add_upsample_layer(const int layer_idx,
                            std::vector<nvinfer1::ITensor*>& fmaps) const
        -> void;

    auto add_yolo_layer(const int layer_idx,
                        std::vector<nvinfer1::ITensor*>& fmaps) const -> void;

    auto operator()() -> void;

    auto operator()(const std::size_t begin, const std::size_t end) -> void;

    auto operator()(const std::size_t at) -> void
    {
      (*this)(at, at + 1);
    }
  };


  auto
  convert_yolo_v4_network_from_darknet(const std::string& trained_model_dir,
                                       const bool is_tiny = true)
      -> HostMemoryUniquePtr;

}  // namespace DO::Sara::TensorRT
