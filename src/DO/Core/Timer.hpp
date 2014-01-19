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

#ifndef DO_CORE_TIMER_HPP
#define DO_CORE_TIMER_HPP

#include <ctime>
#include <iostream>
#ifdef _WIN32
# include <windows.h>
#else
# include <sys/time.h>
#endif

namespace DO {

  //! \ingroup Core
  //! \defgroup Utility Utility
  //! @{

  //! \brief Timer class.
  class Timer
  {
  public: /* interface. */
    //! Default constructor
    Timer()
      : start_(std::clock())
      , elapsed_(0)
    {
    }
    //! Reset the timer to zero.
    void restart()
    {
      start_ = std::clock();
      elapsed_ = 0;
    }
    //! Returns the elapsed time.
    double elapsed()
    {
      elapsed_ = std::clock() - start_;
      return double(elapsed_) / double(CLOCKS_PER_SEC);
    }
    //! Helper function that prints the elapsed time in a friendly manner.
    void print()
    {
      std::cout << "Elapsed time: " << elapsed() << " s" << std::endl;
    }
  private:
    std::clock_t start_; //!< Records the start instant.
    std::clock_t elapsed_; //!< Stores the elapsed time from the start instant.
  };

  //! \brief Timer class with microsecond accuracy.
  class HighResTimer
  {
  public: /* interface. */
    //! Default constructor
    HighResTimer()
      : elapsed_(0)
    {
#ifdef WIN32
      if (!QueryPerformanceFrequency(&frequency_))
      {
        const char *msg = "Failed to initialize high resolution timer!";
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
      }
#endif
    }
    //! Reset the timer to zero.
    void restart()
    {
#ifdef WIN32
      QueryPerformanceCounter(&start_);
#else
      gettimeofday(&start_, NULL);
#endif
    }
    //! Returns the elapsed time in seconds.
    double elapsed()
    {
#ifdef _WIN32
      QueryPerformanceCounter(&end_);
      elapsed_ = static_cast<double>(end_.QuadPart - start_.QuadPart)
               / frequency_.QuadPart;
#else
      gettimeofday(&end_, NULL);
      elapsed_ = (end_.tv_sec - start_.tv_sec);
      elapsed_ += (end_.tv_usec - start_.tv_usec) / 1e6;
#endif
      return elapsed_;
    }
    //! Returns the elapsed time in milliseconds.
    double elapsedMs()
    { return elapsed() * 1e3; }
  private: /* data members. */
#ifdef WIN32
    LARGE_INTEGER frequency_;
    LARGE_INTEGER start_;
    LARGE_INTEGER end_;
#else
    timeval start_;
    timeval end_;
#endif
    double elapsed_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_CORE_TIMER_HPP */