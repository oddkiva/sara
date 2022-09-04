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

#include <drafts/OpenCL/Device.hpp>
#include <drafts/OpenCL/DeviceBuffer.hpp>
#include <drafts/OpenCL/DeviceImage.hpp>
#include <drafts/OpenCL/Error.hpp>
#include <drafts/OpenCL/Program.hpp>

#include <memory>
#include <vector>


namespace DO::Sara {

  //! @addtogroup OpenCL
  //! @{

  class Kernel
  {
  public:
    Kernel() = default;

    ~Kernel()
    {
      if (!_kernel)
        return;
      cl_int err = clReleaseKernel(_kernel);
      if (err < 0)
        std::cerr << format("Error: failed to release kernel! %s",
                            get_error_string(err))
                  << std::endl;
    }

    operator cl_kernel() const
    {
      return _kernel;
    }

    template <typename T>
    bool set_argument(T& arg_name, int arg_pos)
    {
      auto err = clSetKernelArg(_kernel, arg_pos, sizeof(T), &arg_name);

      if (err < 0)
      {
        std::cerr << format("Error: failed to set kernel argument! %s",
                            get_error_string(err))
                  << std::endl;
        return false;
      }

      return true;
    }

    template <typename T>
    bool set_argument(DeviceBuffer<T>& arg_name, int arg_pos)
    {
      cl_mem& mem = arg_name;
      auto err = clSetKernelArg(_kernel, arg_pos, sizeof(cl_mem), &mem);

      if (err < 0)
      {
        std::cerr << format("Error: failed to set kernel argument! %s",
                            get_error_string(err))
                  << std::endl;
        return false;
      }

      return true;
    }

    template <typename T>
    bool set_argument(DeviceImage<T>& arg_name, int arg_pos)
    {
      cl_mem& mem = arg_name;
      auto err = clSetKernelArg(_kernel, arg_pos, sizeof(cl_mem), &mem);

      if (err < 0)
      {
        std::cerr << format("Error: failed to set kernel argument! %s",
                            get_error_string(err))
                  << std::endl;
        return false;
      }

      return true;
    }

    friend bool get_kernels_from_program(std::vector<Kernel>& kernels,
                                         const Program& program)
    {
      auto err = cl_int{};

      // Count the number of kernels to create from the program.
      auto num_kernels = cl_uint{};
      err = clCreateKernelsInProgram(program, 0, nullptr, &num_kernels);
      if (err < 0)
      {
        std::cerr << format("Error: failed to fetch the number of kernels from "
                            "program! %s",
                            get_error_string(err))
                  << std::endl;
        return false;
      }

      // Create the list of cl_kernels.
      auto cl_kernels = std::vector<cl_kernel>(num_kernels);
      err = clCreateKernelsInProgram(program,
                                     static_cast<cl_uint>(cl_kernels.size()),
                                     &cl_kernels[0], nullptr);
      if (err < 0)
      {
        std::cerr << format("Error: failed to fetch the number of kernels from "
                            "program! %s",
                            get_error_string(err))
                  << std::endl;
        return false;
      }

      // Create the list of wrapped kernels.
      kernels.resize(num_kernels);
      for (size_t i = 0; i != cl_kernels.size(); ++i)
        kernels[i]._kernel = cl_kernels[i];

      return true;
    }

  private: /* data members. */
    cl_kernel _kernel = nullptr;
  };

  //! @}

} /* namespace DO::Sara */
