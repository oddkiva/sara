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

#include <stdexcept>
#include <string>
#include <vector>

#include <drafts/OpenCL/Core/OpenCL.hpp>

#include <drafts/OpenCL/Core/Error.hpp>
#include <drafts/OpenCL/Core/Platform.hpp>


namespace DO::Sara {

  //! @addtogroup OpenCL
  //! @{

  class Device
  {
  public:
    cl_device_id id;
    cl_int type;
    std::string name;
    std::string vendor;
    std::string extensions;
    cl_ulong global_memory_size;
    cl_uint address_space;
    cl_bool available;
    cl_bool compiler_available;

    operator cl_device_id() const
    {
      return id;
    }
  };

  std::ostream& operator<<(std::ostream& os, const Device& p)
  {
    const size_t length = std::string("Compiler available  ").size();
    using std::left;
    using std::setw;
    os << left << setw(length) << "ID"
       << ":  " << p.id << "\n";
    os << left << setw(length) << "Name"
       << ":  " << p.name << "\n";
    os << left << setw(length) << "Vendor"
       << ":  " << p.vendor << "\n";
    os << left << setw(length) << "Extensions"
       << ":  " << p.extensions << "\n";
    os << left << setw(length) << "Global mem size"
       << ":  " << p.global_memory_size << "\n";
    os << left << setw(length) << "Address space"
       << ":  " << p.address_space << "\n";
    os << left << setw(length) << "Available"
       << ":  " << p.available << "\n";
    os << left << setw(length) << "Compiler available"
       << ":  " << p.compiler_available << "\n";
    return os;
  }

  template<int _InfoType>
  std::string get_device_string_info(cl_device_id device_id)
  {
    cl_int err;

    size_t buffer_length;
    err = clGetDeviceInfo(device_id, _InfoType, 0, nullptr, &buffer_length);
    if (err < 0)
    {
      std::cerr << format("Error: cannot get device info! %s",
                          get_error_string(err))
                << std::endl;
      return std::string{};
    }

    auto buffer = std::vector<char>{};
    buffer.resize(buffer_length);
    err = clGetDeviceInfo(device_id, _InfoType, buffer_length, &buffer[0],
                          nullptr);
    if (err < 0)
    {
      std::cerr << format("Error: cannot get device info! %s",
                          get_error_string(err))
                << std::endl;
      return std::string{};
    }

    return std::string{buffer.begin(), buffer.end()};
  }


  template<typename T, int InfoType>
  T get_device_info(cl_device_id device_id)
  {
    auto info = T{};

    auto err = clGetDeviceInfo(device_id, InfoType, sizeof(T), &info, nullptr);
    if (err < 0)
    {
      std::cerr << format("Error: cannot get device info! %s\n",
                          get_error_string(err))
                << std::endl;
    }

    return info;
  }


  std::vector<cl_device_id> to_id(const std::vector<Device>& devices)
  {
    std::vector<cl_device_id> ids;
    for (const auto& device : devices)
      ids.push_back(device);
    return ids;
  }


  std::vector<Device>
  get_devices(const Platform& platform,
              cl_device_type device_type = CL_DEVICE_TYPE_ALL)
  {
    auto err = cl_int{};

    auto num_devices = cl_uint{};
    err = clGetDeviceIDs(platform.id, device_type, 0, nullptr, &num_devices);
    if (err < 0)
    {
      std::cerr
          << format(
                 "Error: cannot get number of devices from platform %p! %s\n",
                 platform.id, get_error_string(err))
          << std::endl;
      return std::vector<Device>{};
    }

    auto device_ids = std::vector<cl_device_id>(num_devices);
    clGetDeviceIDs(platform.id, device_type, num_devices, &device_ids[0],
                   nullptr);

    auto devices = std::vector<Device>(num_devices);
    for (cl_uint i = 0; i < num_devices; ++i)
    {
      const auto& id = device_ids[i];
      auto& device = devices[i];

      device.id = id;
      device.name = get_device_string_info<CL_DEVICE_NAME>(id);
      device.vendor = get_device_string_info<CL_DEVICE_VENDOR>(id);
      device.extensions = get_device_string_info<CL_DEVICE_EXTENSIONS>(id);
      device.global_memory_size =
          get_device_info<cl_ulong, CL_DEVICE_GLOBAL_MEM_SIZE>(id);
      device.address_space =
          get_device_info<cl_uint, CL_DEVICE_ADDRESS_BITS>(id);
      device.available = get_device_info<cl_bool, CL_DEVICE_AVAILABLE>(id);
      device.compiler_available =
          get_device_info<cl_bool, CL_DEVICE_COMPILER_AVAILABLE>(id);
    }

    return devices;
  }

  //! @}

} /* namespace DO::Sara */
