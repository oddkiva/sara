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

#include <DO/Shakti/OpenCL/Error.hpp>

#include <fmt/format.h>

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


namespace DO::Sara {

  //! @addtogroup OpenCL
  //! @{

  //! @brief OpenCL platform structure.
  struct Platform
  {
    cl_platform_id id;
    std::string name;
    std::string vendor;
    std::string version;
    std::string profile;
    std::string extensions;

    friend inline auto operator<<(std::ostream& os, const Platform& p)
        -> std::ostream&
    {
      const auto length = static_cast<int>(std::string("Extensions  ").size());
      using std::left;
      using std::setw;
      os << left << setw(length) << "ID"
         << ":  " << p.id << "\n";
      os << left << setw(length) << "Name"
         << ":  " << p.name << "\n";
      os << left << setw(length) << "Vendor"
         << ":  " << p.vendor << "\n";
      os << left << setw(length) << "Version"
         << ":  " << p.version << "\n";
      os << left << setw(length) << "Profile"
         << ":  " << p.profile << "\n";
      os << left << setw(length) << "Extensions"
         << ":  " << p.extensions << "\n";
      return os;
    }
  };

  //! @brief Read the OpenCL platform.
  template <int _InfoType>
  inline auto get_platform_info(cl_platform_id platform_id) -> std::string
  {
    cl_int err;

    size_t buffer_length;
    err = clGetPlatformInfo(platform_id, _InfoType, 0, nullptr, &buffer_length);
    if (err < 0)
      throw std::runtime_error(fmt::format(
          "Error: cannot get platform info! {}", get_error_string(err)));

    std::vector<char> buffer;
    buffer.resize(buffer_length);
    err = clGetPlatformInfo(platform_id, _InfoType, buffer_length, &buffer[0],
                            nullptr);
    if (err < 0)
      throw std::runtime_error(fmt::format(
          "Error: cannot get platform info! {}", get_error_string(err)));

    return std::string(buffer.begin(), buffer.end());
  }


  inline auto get_platforms() -> std::vector<Platform>
  {
    cl_int err;

    cl_uint num_platforms;
    err = clGetPlatformIDs(1, nullptr, &num_platforms);
    if (err < 0)
      throw std::runtime_error{fmt::format(
          "Error: cannot get number of platforms! {}", get_error_string(err))};

    std::vector<cl_platform_id> platform_ids(num_platforms);
    err = clGetPlatformIDs(num_platforms, &platform_ids[0], nullptr);
    if (err < 0)
      throw std::runtime_error{
          fmt::format("Error: cannot get the list of platforms! {}",
                      get_error_string(err))};


    std::vector<Platform> platforms(num_platforms);
    for (cl_uint i = 0; i < num_platforms; ++i)
    {
      const auto& id = platform_ids[i];
      auto& platform = platforms[i];

      platform.id = id;
      platform.name = get_platform_info<CL_PLATFORM_NAME>(id);
      platform.vendor = get_platform_info<CL_PLATFORM_VENDOR>(id);
      platform.version = get_platform_info<CL_PLATFORM_VERSION>(id);
      platform.profile = get_platform_info<CL_PLATFORM_PROFILE>(id);
      platform.extensions = get_platform_info<CL_PLATFORM_EXTENSIONS>(id);
    }

    return platforms;
  }

  //! @}

} /* namespace DO::Sara */
