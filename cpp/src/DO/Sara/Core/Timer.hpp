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
    double _start;
#if defined(_WIN32) || defined(_WIN32_WCE)
    double _frequency;
#endif
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */
