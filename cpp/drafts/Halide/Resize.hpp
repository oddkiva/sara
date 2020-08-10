// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <drafts/Halide/Utilities.hpp>

#include "shakti_enlarge.h"
#include "shakti_reduce_32f.h"
#include "shakti_scale_32f.h"


namespace DO { namespace Shakti { namespace HalideBackend {

  auto scale(Sara::ImageView<float>& src, Sara::ImageView<float>& dst)
  {
    auto src_tensor_view = tensor_view(src).reshape(
        Eigen::Vector4i{1, 1, src.height(), src.width()});
    auto dst_tensor_view = tensor_view(dst).reshape(
        Eigen::Vector4i{1, 1, dst.height(), dst.width()});

    auto src_buffer = as_runtime_buffer(src_tensor_view);
    auto dst_buffer = as_runtime_buffer(dst_tensor_view);

    src_buffer.set_host_dirty();
    shakti_scale_32f(src_buffer, dst.width(), dst.height(), dst_buffer);
    dst_buffer.copy_to_host();
  }

  auto reduce(Sara::ImageView<float>& src, Sara::ImageView<float>& dst)
  {
    auto src_tensor_view = tensor_view(src).reshape(
        Eigen::Vector4i{1, 1, src.height(), src.width()});
    auto dst_tensor_view = tensor_view(dst).reshape(
        Eigen::Vector4i{1, 1, dst.height(), dst.width()});

    auto src_buffer = as_runtime_buffer(src_tensor_view);
    auto dst_buffer = as_runtime_buffer(dst_tensor_view);

    src_buffer.set_host_dirty();
    shakti_reduce_32f(src_buffer, dst.width(), dst.height(), dst_buffer);
    dst_buffer.copy_to_host();
  }

  auto enlarge(Sara::ImageView<float>& src, Sara::ImageView<float>& dst)
  {
    auto src_buffer = as_runtime_buffer_3d(src);
    auto dst_buffer = as_runtime_buffer_3d(dst);

    src_buffer.set_host_dirty();
    shakti_enlarge(src_buffer, src_buffer.width(), src_buffer.height(),
                   dst_buffer.width(), dst_buffer.height(), dst_buffer);
    dst_buffer.copy_to_host();
  }

  auto enlarge(Sara::ImageView<Sara::Rgb32f>& src,
               Sara::ImageView<Sara::Rgb32f>& dst)
  {
    auto src_buffer = as_interleaved_runtime_buffer(src);
    auto dst_buffer = as_interleaved_runtime_buffer(dst);

    src_buffer.set_host_dirty();
    shakti_enlarge(src_buffer, src_buffer.width(), src_buffer.height(),
                   dst_buffer.width(), dst_buffer.height(), dst_buffer);
    dst_buffer.copy_to_host();
  }

}}}  // namespace DO::Shakti::HalideBackend
