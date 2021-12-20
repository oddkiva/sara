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

#include <DO/Sara/ImageProcessing/Resize.hpp>

#ifdef DO_SARA_USE_HALIDE
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include "shakti_enlarge_cpu.h"
#include "shakti_reduce_32f_cpu.h"
#include "shakti_scale_32f_cpu.h"
#endif


namespace DO::Sara {

  auto scale(const ImageView<float>& src, ImageView<float>& dst) -> void
  {
    auto src_tensor_view = tensor_view(src).reshape(
        Eigen::Vector4i{1, 1, src.height(), src.width()});
    auto dst_tensor_view = tensor_view(dst).reshape(
        Eigen::Vector4i{1, 1, dst.height(), dst.width()});

    auto src_buffer = Shakti::Halide::as_runtime_buffer(src_tensor_view);
    auto dst_buffer = Shakti::Halide::as_runtime_buffer(dst_tensor_view);

    shakti_scale_32f_cpu(src_buffer, dst_buffer.width(), dst_buffer.height(),
                         dst_buffer);
  }

  auto downscale(const ImageView<float>& src, int fact) -> Image<float>
  {
    auto timer = Timer{};
    timer.restart();

    auto dst = Image<float>{(src.sizes() / fact).eval()};
    scale(src, dst);

    const auto elapsed = timer.elapsed_ms();
    SARA_DEBUG << "[CPU Halide Downscale][" << src.sizes().transpose()
               << "] " << elapsed << " ms" << std::endl;

    return dst;
  }

  auto enlarge(const ImageView<float>& src, ImageView<float>& dst) -> void
  {
    auto src_tensor_view = tensor_view(src).reshape(
        Eigen::Vector4i{1, 1, src.height(), src.width()});
    auto dst_tensor_view = tensor_view(dst).reshape(
        Eigen::Vector4i{1, 1, dst.height(), dst.width()});

    auto src_buffer = Shakti::Halide::as_runtime_buffer(src_tensor_view);
    auto dst_buffer = Shakti::Halide::as_runtime_buffer(dst_tensor_view);

    shakti_enlarge_cpu(src_buffer,                               //
                       src_buffer.width(), src_buffer.height(),  //
                       dst_buffer.width(), dst_buffer.height(),  //
                       dst_buffer);
  }

  auto enlarge(const ImageView<Rgb32f>& src, ImageView<Rgb32f>& dst) -> void
  {
    auto& src_non_const = const_cast<ImageView<Rgb32f>&>(src);
    auto src_buffer = ::Halide::Runtime::Buffer<float>::make_interleaved(
        reinterpret_cast<float*>(src_non_const.data()), src.width(),
        src.height(), 3);
    auto dst_buffer = ::Halide::Runtime::Buffer<float>::make_interleaved(
        reinterpret_cast<float*>(dst.data()), dst.width(), dst.height(), 3);
    src_buffer.add_dimension();
    dst_buffer.add_dimension();

    shakti_enlarge_cpu(src_buffer,                 //
                       src.width(), src.height(),  //
                       dst.width(), dst.height(),  //
                       dst_buffer);
  }


}  // namespace DO::Sara
