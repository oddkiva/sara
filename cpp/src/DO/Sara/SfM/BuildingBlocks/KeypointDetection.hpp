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

#include <DO/Sara/Defines.hpp>

#include <string>


namespace DO::Sara {

  //! @addtogroup SfM
  //! @{

  //! @{
  //! @brief Keypoint detection.
  auto detect_keypoints(const std::string& dirpath,
                        const std::string& h5_filepath, bool overwrite) -> void;

  auto read_keypoints(const std::string& dirpath,
                      const std::string& h5_filepath) -> void;
  //! @}

  //! @}

} /* namespace DO::Sara */
