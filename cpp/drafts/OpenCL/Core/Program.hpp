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

#include <drafts/OpenCL/Core/Context.hpp>
#include <drafts/OpenCL/Core/Device.hpp>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include <fstream>
#include <streambuf>
#include <vector>


namespace DO::Sara {

  class Program
  {
  public:
    Program(Context& context, const Device& device)
      : _context(context)
    {
      _devices.push_back(device);
    }

    ~Program()
    {
      auto err = clReleaseProgram(_program);
      if (err < 0)
        std::cerr << "Error: failed to release OpenCL program!" << std::endl;
    }

    operator cl_program() const
    {
      return _program;
    }

    void set_devices(const std::vector<Device>& devices)
    {
      _devices = devices;
    }

    void set_compiler_options(const std::string& compilation_options)
    {
      _compilation_options = compilation_options;
    }

    bool create_from_source(const std::string& source)
    {
      auto err = cl_int{};
      const auto *source_data = &source[0];
      auto source_size = source.size();
      _program = clCreateProgramWithSource(
        _context, 1, &source_data, &source_size, &err);

      if (err < 0)
      {
        std::cerr << format("Error: failed to create program from source: %s",
                            source.c_str()) << std::endl;
        return false;
      }

      return true;
    }

    bool create_from_file(const std::string& filepath)
    {
      // Read source fle.
      std::ifstream file(filepath.c_str());
      if (!file.is_open())
      {
        std::cerr << format("Error: cannot open file: %s", filepath.c_str())
                  << std::endl;
        return false;
      }

      auto source = std::string{};
      file.seekg(0, std::ios::end);   
      source.reserve(file.tellg());

      file.seekg(0, std::ios::beg);
      source.assign(std::istreambuf_iterator<char>(file),
                    std::istreambuf_iterator<char>());

      return create_from_source(source);
    }

    bool build()
    {
      auto device_ids = to_id(_devices);
      cl_int err = clBuildProgram(
        _program, static_cast<cl_uint>(device_ids.size()), &device_ids[0],
        nullptr, nullptr, nullptr);

      if (err < 0)
      {
        std::cerr << format("Error: failed to build OpenCL program:\n%s",
                            get_build_logs().c_str()) << std::endl;
        return false;
      }

      return true;
    }

    std::string get_build_logs(const Device& device) const
    {
      cl_int err;

      size_t build_log_size;
      err = clGetProgramBuildInfo(_program, device, CL_PROGRAM_BUILD_LOG, 0,
                                  NULL, &build_log_size);
      if (err < 0)
        throw std::runtime_error("Error: failed to get build log size!");

      std::string build_log;
      build_log.resize(build_log_size);
      err = clGetProgramBuildInfo(_program, device, CL_PROGRAM_BUILD_LOG,
                                  build_log.size(), &build_log[0], nullptr);
      if (err < 0)
        throw std::runtime_error("Error: failed to get build log!");

      return build_log;
    }

    std::string get_build_logs() const
    {
      auto build_log = std::string{};
      for (const auto& device : _devices)
      {
        std::string device_build_log = get_build_logs(device);
        build_log += device_build_log;
        build_log += "\n\n";
      }
      return build_log;
    }

  private:
    Context& _context;
    cl_program _program = nullptr;
    std::string _compilation_options;
    std::vector<Device> _devices;
    std::vector<std::string> _binaries;
  };

} /* namespace DO::Sara */
