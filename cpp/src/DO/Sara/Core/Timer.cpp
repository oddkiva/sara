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

#include <DO/Sara/Core/Timer.hpp>

#include <stdexcept>


namespace DO { namespace Sara {

  Timer::Timer()
  {
  }

  void Timer::restart()
  {
    _start = Clock::now();
  }

  double Timer::elapsed()
  {
    const auto _end = Clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start)
            .count() *
        1e-9;
    return elapsed;
  }

  double Timer::elapsed_ms()
  {
    const auto _end = Clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(_end - _start)
            .count() *
        1e-6;
    return elapsed;
  }

}}  // namespace DO::Sara
