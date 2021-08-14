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

#include <DO/Sara/Defines.hpp>

#include <chrono>


namespace DO { namespace Sara {

  //! @ingroup Core
  //! @defgroup Utility Utility
  //! @{

  //! @brief Timer class with microsecond accuracy.
  class Timer
  {
  public: /* interface. */
    //! Default constructor
    DO_SARA_EXPORT
    Timer();

    //! Reset the timer to zero.
    DO_SARA_EXPORT
    void restart();

    //! Returns the elapsed time in seconds.
    DO_SARA_EXPORT
    double elapsed();

    //! Returns the elapsed time in milliseconds.
    DO_SARA_EXPORT
    double elapsed_ms();

  private: /* data members. */
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::high_resolution_clock::time_point _start =
        std::chrono::high_resolution_clock::now();
  };

  //! @}

}}  // namespace DO::Sara
