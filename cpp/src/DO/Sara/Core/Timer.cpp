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

#ifdef _WIN32
# include <windows.h>
#else
# include <sys/time.h>
#endif

#include <DO/Sara/Core/Timer.hpp>


namespace DO { namespace Sara {

  Timer::Timer()
  {
#ifdef _WIN32
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq))
    {
      auto msg = "Failed to initialize high resolution timer!";
      throw std::runtime_error{msg};
    }
    _frequency = static_cast<double>(freq.QuadPart);
#endif
    restart();
  }

  void Timer::restart()
  {
#ifdef _WIN32
    LARGE_INTEGER _li_start;
    QueryPerformanceCounter(&_li_start);
    _start = static_cast<double>(_li_start.QuadPart);
#else
    timeval start;
    gettimeofday(&start, nullptr);
    _start = start.tv_sec + start.tv_usec * 1e-6;
#endif
  }

  double Timer::elapsed()
  {
#ifdef _WIN32
    LARGE_INTEGER _end;
    QueryPerformanceCounter(&_end);
    return (static_cast<double>(_end.QuadPart) - _start) / _frequency;
#else
    timeval end;
    gettimeofday(&end, nullptr);
    double _end = end.tv_sec + end.tv_usec * 1e-6;
    return _end - _start;
#endif
  }

  double Timer::elapsed_ms()
  {
    return elapsed() * 1000.;
  }

} /* namespace Sara */
} /* namespace DO */
