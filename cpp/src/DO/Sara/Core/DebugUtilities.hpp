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
// To avoid compilation error with Eigen
#if (defined(_WIN32) || defined(_WIN32_WCE)) && !defined(NOMINMAX)
#  define NOMINMAX
#endif

#include <termcolor/termcolor.hpp>

#include <iostream>
#include <string>

#define __FILENAME__                                                           \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef __APPLE__
#  define SARA_DEBUG                                                           \
    std::cout << "🧭[" << __FILENAME__ << "]"                                \
              << "📑[" << __FUNCTION__ << ":" << __LINE__ << "]🎶 "

#  define SARA_CHECK(x)                                                        \
    std::cout << "🧭[" << __FILENAME__ << "]"                                \
              << "📑[" << __FUNCTION__ << ":" << __LINE__ << "]🎶 " << #x        \
              << " = " << x << std::endl
#else
#  define SARA_DEBUG                                                           \
    std::cout << termcolor::bold << termcolor::green << "🧭["                \
              << __FILENAME__ << "]"                                           \
              << "📑" << termcolor::red << "[" << __FUNCTION__ << ":"           \
              << __LINE__ << "]🎶 " << termcolor::reset

#  define SARA_CHECK(x)                                                        \
    std::cout << termcolor::bold << termcolor::green << "🧭["                \
              << __FILENAME__ << "]"                                           \
              << "📑" << termcolor::red << "[" << __FUNCTION__ << ":"           \
              << __LINE__ << "]🎶" << termcolor::reset << "\n"                  \
              << #x << " = " << x << std::endl
#endif


namespace DO { namespace Sara {

  //! @ingroup Utility
  //! @brief Output program stage description on console.
  inline void print_stage(const std::string& stageName)
  {
    std::cout << std::endl;
    std::cout << "// "
                 "============================================================="
                 "========== //"
              << std::endl;
    std::cout << "// " << stageName << std::endl;
  }

  //! @ingroup Utility
  //! @brief Wait for return key on the console.
  inline void wait_return_key()
  {
    std::cout << "Press RETURN key to continue...";
    std::cin.ignore();
  }

}}  // namespace DO::Sara
