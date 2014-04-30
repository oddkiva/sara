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

#include <gtest/gtest.h>
#include <DO/Core/Timer.hpp>
#include <DO/Core/DebugUtilities.hpp>
#include <chrono>
#include <thread>

using namespace DO;
using namespace std;

inline void milliSleep(unsigned milliseconds)
{
  chrono::milliseconds duration(milliseconds);
  this_thread::sleep_for(duration);
}

TEST(DO_Core_Test, testTimer)
{
  Timer timer;
  double elapsedTimeMs;
  double elapsedTimeS;
  unsigned sleepTimeMs = 580;

  timer.restart();
  milliSleep(sleepTimeMs);
  elapsedTimeMs =  timer.elapsedMs();
  EXPECT_NEAR(elapsedTimeMs, sleepTimeMs, 100);

  timer.restart();
  milliSleep(sleepTimeMs);
  elapsedTimeS = timer.elapsed();
  EXPECT_NEAR(elapsedTimeS, sleepTimeMs/1e3, 2e-3);
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}
