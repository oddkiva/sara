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

#include <DO/Core/Timer.hpp>
#ifdef _WIN32
# include <windows.h>
#else
# include <sys/time.h>
#endif

namespace DO {

  Timer::Timer()
    : start_(std::clock())
    , elapsed_(0) 
  {
  }
  
  void Timer::restart()
  {
    start_ = std::clock();
    elapsed_ = 0;
  }

  double Timer::elapsed()
  {
    elapsed_ = static_cast<double>(std::clock()) - start_;
    elapsed_ /= CLOCKS_PER_SEC;
    return elapsed_;
  }

  HighResTimer::HighResTimer()
    : elapsed_(0)
  {
#ifdef WIN32
    LARGE_INTEGER freq;
    if (!QueryPerformanceFrequency(&freq))
    {
      const char *msg = "Failed to initialize high resolution timer!";
      std::cerr << msg << std::endl;
      throw std::runtime_error(msg);
    }
    frequency_ = static_cast<double>(freq.QuadPart);
#endif
  }
  //! Reset the timer to zero.
  void HighResTimer::restart()
  {
#ifdef WIN32
    LARGE_INTEGER li_start_;
    QueryPerformanceCounter(&li_start_);
    start_ = static_cast<double>(li_start_.QuadPart);
#else
    timeval start;
    gettimeofday(&start, NULL);
    start_ = start.tv_sec * 1e3 + start.tv_usec * 1e-3;
#endif
  }
  //! Returns the elapsed time in seconds.
  double HighResTimer::elapsed()
  {
#ifdef _WIN32
    LARGE_INTEGER end_;
    QueryPerformanceCounter(&end_);
    elapsed_ = (static_cast<double>(end_.QuadPart) - start_) / frequency_;
#else
    timeval end_;
    gettimeofday(&end_, NULL);
    double end_ = end_.tv_sec * 1e3 + end_.tv_usec * 1e-3;
    elapsed_ = end_ - start_;
#endif
    return elapsed_;
  }

  double HighResTimer::elapsedMs()
  {
    return elapsed() * 1000.; 
  }

} /* namespace DO */