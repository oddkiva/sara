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

#ifdef WIN32
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
  public:
    //! Default constructor
    Timer()
      : start_(std::clock())
      , elapsed_(0) {}
    
    //! Reset the timer to zero.
    void restart()
    {
      start_ = std::clock();
      elapsed_ = 0;
    }

    //! Returns the elapsed time.
    double elapsed()
    {
      elapsed_ = static_cast<double>(std::clock()) - start_;
      elapsed_ /= CLOCKS_PER_SEC;
      return elapsed_;
    }

    //! Helper function that prints the elapsed time in a friendly manner.
    void print()
    {
      std::cout << "Elapsed time: " << elapsed() << " s" << std::endl;
    }
  
  private:
    std::clock_t start_; //!< Records the start instant.
    double elapsed_; //!< Stores the elapsed time from the start instant.
  };

  //! \brief Timer class with microsecond accuracy.
  class HighResTimer
  {
  public:
    HighResTimer ()
      : elapsed_(0)
    {
#ifdef WIN32
      QueryPerformanceCounter(&frequency_);
#endif
    }
    void restart()
    {
#ifdef WIN32
      QueryPerformanceCounter(&start_);
#else
      gettimeofday(&start_, NULL);
#endif
      elapsed_ = 0;
    }

    double elapsedMs()
    {
#ifdef WIN32
      QueryPerformanceCounter(&end_);
      elapsed_ = (end_.QuadPart - start_.QuadPart)*1000.0 / frequency_.QuadPart;
#else
      gettimeofday(&end_, NULL);
      elapsed_ = (end_.tv_sec - start_.tv_sec) * 1000.0;
      elapsed_ += (end_.tv_usec - start_.tv_usec) * 1000.0;
#endif
      return elapsed_;
    }

#ifdef WIN32
    LARGE_INTEGER start_, end_;
    LARGE_INTEGER frequency_;
#else
    timeval start_, end_;
#endif

    double elapsed_;
  };

  //! @}

} /* namespace DO */