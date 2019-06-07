// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <iostream>
#include <string>


#define SARA_DEBUG std::cout << "[" << __FUNCTION__ << ":" << __LINE__ << "] "
#define SARA_CHECK(x)                                                          \
  std::cout << "[" << __FUNCTION__ << ":" << __LINE__ << "] " << #x << " = "   \
            << x << std::endl


namespace DO { namespace Sara {

  //! @ingroup Utility
  //! @brief Outputting program stage description on console.
  inline void print_stage(const std::string& stageName)
  {
    std::cout << std::endl;
    std::cout << "// ======================================================================= //" << std::endl;
    std::cout << "// " << stageName << std::endl;
  }

  //! @ingroup Utility
  //! @brief Wait for return key on the console.
  inline void wait_return_key()
  {
    std::cout << "Press RETURN key to continue...";
    std::cin.ignore();
  }

} /* namespace Sara */
} /* namespace DO */
