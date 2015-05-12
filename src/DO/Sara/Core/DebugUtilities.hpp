// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_DEBUG_UTILITIES_HPP
#define DO_DEBUG_UTILITIES_HPP

#include <iostream>

#define CHECK(x) std::cout << #x << " = " << x << std::endl


namespace DO {

  //! \ingroup Utility
  //! \brief Outputting program stage description on console.
  inline void print_stage(const std::string& stageName)
  {
    std::cout << std::endl;
    std::cout << "// ======================================================================= //" << std::endl;
    std::cout << "// " << stageName << std::endl;
  }

  //! \ingroup Utility
  //! \brief Wait for return key on the console.
  inline void wait_return_key()
  {
    std::cout << "Press RETURN key to continue...";
    std::cin.ignore();
  }

} /* namespace DO */


#endif /* DO_DEBUG_UTILITIES_HPP */
