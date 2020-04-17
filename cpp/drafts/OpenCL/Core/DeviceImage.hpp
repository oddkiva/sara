// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <drafts/OpenCL/Core/Program.hpp>

#include <array>


namespace DO::Sara {

  //! @addtogroup OpenCL
  //! @{

  //! @brief OpenCL pixel traits structure.
  template <typename T>
  struct CLPixelTraits;

  template <>
  struct CLPixelTraits<float>
  {
    using value_type = float;

    enum
    {
      PixelType = CL_FLOAT,
      ColorSpace = CL_INTENSITY
    };
  };

  //! @brief OpenCL device image class.
  template <typename T, int N = 2>
  class DeviceImage
  {
  public:
    using pixel_type = T;
    using pixel_traits = CLPixelTraits<T>;

    enum
    {
      PixelType = pixel_traits::PixelType,
      ColorSpace = pixel_traits::ColorSpace
    };

  public:
    inline DeviceImage()
    {
      _sizes.fill(0);
    }

    DeviceImage(Context& context, size_t width, size_t height, const T* data,
                cl_mem_flags flags)
      : _sizes{width, height}
    {
      auto err = cl_int{};
      auto image_format = cl_image_format{ColorSpace, PixelType};
      _buffer = clCreateImage2D(context, flags, &image_format, _sizes[0],
                                _sizes[1], 0, (void*) data, &err);
      if (err < 0)
        throw std::runtime_error(
            format("Error: failed to allocate buffer in device memory! %s\n",
                   get_error_string(err)));
    }

    DeviceImage(Context& context, size_t width, size_t height, size_t depth,
                const T* data, cl_mem_flags flags)
      : _sizes{width, height, depth}
    {
      auto err = cl_int{};
      auto image_format = cl_image_format{ColorSpace, PixelType};
      _buffer = clCreateImage3D(context, flags, &image_format, _sizes[0],
                                _sizes[1], _sizes[2], 0, (void*) data, &err);
      if (err < 0)
        throw std::runtime_error(
            format("Error: failed to allocate buffer in device memory! %s\n",
                   get_error_string(err)));
    }

    ~DeviceImage()
    {
      auto err = clReleaseMemObject(_buffer);
      if (err < 0)
        std::cerr << format(
                         "Error: failed to release buffer in device memory! %s",
                         get_error_string(err))
                  << std::endl;
    }

    inline operator cl_mem&()
    {
      return _buffer;
    }

    inline size_t width() const
    {
      return _sizes[0];
    }

    inline size_t height() const
    {
      return _sizes[1];
    }

    inline size_t depth() const
    {
      return _sizes[2];
    }

  private:
    std::array<size_t, N> _sizes;
    cl_mem _buffer = nullptr;
  };

  //! @}

} /* namespace DO::Sara */
