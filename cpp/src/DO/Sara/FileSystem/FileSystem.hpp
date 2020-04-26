// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <string>
#include <vector>


namespace DO::Sara {

  /*!
   *  @addtogroup FileSystem
   *  @{
   */

  //! @brief Return the basename of a file path.
  DO_SARA_EXPORT
  auto basename(const std::string& filepath) -> std::string;

  //! @brief Minimal subset of command line like API.
  //! @{
  DO_SARA_EXPORT
  auto mkdir(const std::string& dirpath) -> void;

  DO_SARA_EXPORT
  auto cp(const std::string& from, const std::string& to) -> void;

  DO_SARA_EXPORT
  auto ls(const std::string& dirpath, const std::string& ext_filter)
      -> std::vector<std::string>;
  //! @}

  //! @}

} /* namespace DO::Sara */
