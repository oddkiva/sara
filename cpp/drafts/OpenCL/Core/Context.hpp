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

#include <DO/Sara/Core/StringFormat.hpp>

#include <drafts/OpenCL/Core/Device.hpp>

#include <vector>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif


namespace DO::Sara {

  //! @addtogroup OpenCL
  //! @{

  class Context
  {
  public:
    Context(const Device& device)
    {
      auto err = cl_int{};
      _context = clCreateContext(nullptr, 1, &device.id, nullptr, nullptr, &err);
      if (err < 0)
        std::cerr << format("Error: failed to create context from device: %d! %s\n",
                            device.id, get_error_string(err)) << std::endl;

      err = clGetContextInfo(_context, CL_CONTEXT_REFERENCE_COUNT,
                             sizeof(_ref_count), &_ref_count, nullptr);
    }

    ~Context()
    {
      auto err = clReleaseContext(_context);
      if (err < 0)
        std::cerr << format("Error: failed to release OpenCL program! %s\n",
                            get_error_string(err)) << std::endl;
    }

    operator cl_context() const
    {
      return _context;
    }

    template <typename T>
    void push_property(cl_uint key, T value)
    {
      _properties.push_back(key);
      _properties.push_back(reinterpret_cast<cl_context_properties>(value));
    }

  private:
    cl_context _context = nullptr;
    cl_uint _ref_count = 0;
    std::vector<cl_context_properties> _properties;
  };

  //! @}

} /* namespace DO::Sara */
