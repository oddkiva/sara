// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/Image/Image.hpp>
#include <DO/Sara/Core/Pixel/Typedefs.hpp>

#include <DO/Shakti/Cuda/Utilities.hpp>

#include <cuda.h>

#include <memory>


namespace DriverApi {

  auto init() -> void;

  auto get_device_count() -> int;

  struct CudaContext
  {
    CUcontext cuda_context{0};
    CUdevice cuda_device{0};
    int gpu_id{-1};

    CudaContext(int gpu_id_ = 0);

    CudaContext(CudaContext&& other);

    CudaContext(const CudaContext&) = delete;

    ~CudaContext();

    auto make_current() -> void;

    operator CUcontext() const
    {
      return cuda_context;
    }
  };

  struct DeviceBgraBuffer
  {
    int width{};
    int height{};
    CUdeviceptr data{0};

    DeviceBgraBuffer(int width_, int height_);

    ~DeviceBgraBuffer();

    auto to_host(DO::Sara::ImageView<DO::Sara::Bgra8>&) const -> void;
  };

}  // namespace DriverApi


namespace DO { namespace Shakti {

  class VideoStream
  {
  public:
    VideoStream(const std::string& video_filepath,
                const DriverApi::CudaContext& context);

    auto width() const -> int;

    auto height() const -> int;

    auto decode() -> bool;

    auto read(DriverApi::DeviceBgraBuffer&) -> bool;

  private:
    struct Impl;
    struct ImplDeleter
    {
      void operator()(const Impl*) const;
    };
    std::unique_ptr<Impl, ImplDeleter> _impl;
  };

}}  // namespace DO::Shakti


auto show_decoder_capability() -> void;
