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

//! TODO: make better testing of the tree...

#include "gtest/gtest.h"
#include <DO/Core.hpp>
#include <iostream>
#include <list>
#include <utility>
#ifdef _WIN32
# include <windows.h>
#else
# include <unistd.h>
#endif

using namespace DO;
using namespace std;

inline void wait(unsigned milliseconds)
{
#ifdef _WIN32
  Sleep(milliseconds);
#else
  usleep(milliseconds*1000);
#endif
}

TEST(DO_Core_Test,  testTimer)
{
  Timer timer;
  HighResTimer hrTimer;
  double elapsed;

  hrTimer.restart();
  wait(1000);
  elapsed =  hrTimer.elapsedMs();
  EXPECT_NEAR(elapsed, 1000., 1);

  timer.restart();
  wait(1000);
  elapsed = timer.elapsed();
  EXPECT_NEAR(elapsed, 1, 1e-3);
}

int main(int argc, char** argv) 
{

  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}
