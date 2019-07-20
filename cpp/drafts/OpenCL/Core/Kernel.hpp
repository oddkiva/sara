#pragma once

#include <drafts/OpenCL/Core/Device.hpp>
#include <drafts/OpenCL/Core/DeviceBuffer.hpp>
#include <drafts/OpenCL/Core/DeviceImage.hpp>
#include <drafts/OpenCL/Core/Error.hpp>
#include <drafts/OpenCL/Core/Program.hpp>

#include <memory>
#include <vector>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif


namespace DO::Sara {

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
        std::cerr << format(
          "Error: failed to fetch the number of kernels from program! %s",
          get_error_string(err)) << std::endl;
        return false;
      }

      // Create the list of cl_kernels.
      auto cl_kernels = std::vector<cl_kernel>(num_kernels);
      err = clCreateKernelsInProgram(
        program, static_cast<cl_uint>(cl_kernels.size()), &cl_kernels[0],
        nullptr);
      if (err < 0)
      {
        std::cerr << format(
          "Error: failed to fetch the number of kernels from program! %s",
          get_error_string(err)) << std::endl;
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

} /* namespace DO::Sara */
