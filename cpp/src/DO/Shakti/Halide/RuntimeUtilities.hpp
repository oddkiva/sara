// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Shakti/Halide/MyHalide.hpp>


namespace DO::Shakti::Halide {

  template <typename T>
  inline auto as_runtime_buffer(std::vector<T>& v)
  {
    return ::Halide::Runtime::Buffer<T>(v.data(), static_cast<int>(v.size()));
  }

  template <typename T>
  inline auto as_runtime_buffer(const Sara::ImageView<T>& image)
  {
    return ::Halide::Runtime::Buffer<T>(const_cast<T*>(image.data()),
                                        image.width(), image.height());
  }

  template <typename T, typename ColorSpace>
  inline auto as_interleaved_runtime_buffer(
      const Sara::ImageView<Sara::Pixel<T, ColorSpace>>& image)
  {
    auto& image_non_const =
        const_cast<Sara::ImageView<Sara::Pixel<T, ColorSpace>>&>(image);
    return ::Halide::Runtime::Buffer<T>::make_interleaved(
        reinterpret_cast<T*>(image_non_const.data()),  //
        image.width(),                                 //
        image.height(),                                //
        Sara::Pixel<T, ColorSpace>::num_channels()     //
    );
  }

  template <typename T>
  inline auto as_runtime_buffer(const Sara::TensorView_<T, 2>& hw_tensor)
  {
    return ::Halide::Runtime::Buffer<T>(const_cast<T*>(hw_tensor.data()),
                                        hw_tensor.size(1), hw_tensor.size(0));
  }

  template <typename T>
  inline auto as_runtime_buffer(const Sara::TensorView_<T, 3>& chw_tensor)
  {
    return ::Halide::Runtime::Buffer<T>(const_cast<T*>(chw_tensor.data()),
                                        chw_tensor.size(2), chw_tensor.size(1),
                                        chw_tensor.size(0));
  }

  template <typename T>
  inline auto as_runtime_buffer(const Sara::TensorView_<T, 4>& nchw_tensor)
  {
    return ::Halide::Runtime::Buffer<T>(
        const_cast<T*>(nchw_tensor.data()), nchw_tensor.size(3),
        nchw_tensor.size(2), nchw_tensor.size(1), nchw_tensor.size(0));
  }

  template <typename T>
  inline auto as_runtime_buffer_3d(Sara::ImageView<T>& image)
  {
    static constexpr auto num_channels = Sara::PixelTraits<T>::num_channels;
    return ::Halide::Runtime::Buffer<T>(image.data(), image.width(),
                                        image.height(), num_channels);
  }

  template <typename T>
  auto as_runtime_buffer_4d(const Sara::ImageView<T>& src)
      -> ::Halide::Runtime::Buffer<T>
  {
    auto src_non_const = const_cast<Sara::ImageView<T>&>(src);
    auto src_tensor_view =
        tensor_view(src_non_const)
            .reshape(Eigen::Vector4i{1, 1, src.height(), src.width()});
    return as_runtime_buffer(src_tensor_view);
  }

}  // namespace DO::Shakti::Halide
