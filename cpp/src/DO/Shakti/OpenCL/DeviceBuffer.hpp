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

#include <DO/Shakti/OpenCL/Program.hpp>


namespace DO::Sara {

  //! @addtogroup OpenCL
  //! @{

  template <typename T>
  class DeviceBuffer
  {
  public:
    inline DeviceBuffer() = default;

    inline DeviceBuffer(Context& context, T* data, size_t size,
                        cl_mem_flags flags = CL_MEM_READ_WRITE)
      : _data(data)
      , _size(size)
      , _buffer(nullptr)
    {
      auto err = cl_int{};
      _buffer = clCreateBuffer(context, flags, size * sizeof(T), _data, &err);
      if (err != CL_SUCCESS)
        throw std::runtime_error{fmt::format(
            "Error: failed to allocate buffer in device memory! {}\n",
            get_error_string(err))};
    }

    inline ~DeviceBuffer()
    {
      const auto err = clReleaseMemObject(_buffer);
      if (err != CL_SUCCESS)
        std::cerr << fmt::format(
                         "Error: failed to release buffer in device memory! {}",
                         get_error_string(err))
                  << std::endl;
    }

    inline operator cl_mem&()
    {
      return _buffer;
    }

    inline size_t size() const
    {
      return _size;
    }

  private:
    T* _data = nullptr;
    size_t _size = 0;
    cl_mem _buffer = nullptr;
  };

  //! @}

} /* namespace DO::Sara */
