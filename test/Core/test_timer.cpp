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

inline void millisleep(unsigned milliseconds)
{
  chrono::milliseconds duration(milliseconds);
  this_thread::sleep_for(duration);
}

TEST(DO_Core_Test, testTimer)
{
  Timer timer;
  double elapsed_milliseconds;
  double elapsed_seconds;
  unsigned sleep_milliseconds = 580;

  timer.restart();
  millisleep(sleep_milliseconds);
  elapsed_milliseconds =  timer.elapsedMs();
  EXPECT_NEAR(elapsed_milliseconds, sleep_milliseconds, 100);

  timer.restart();
  millisleep(sleep_milliseconds);
  elapsed_seconds = timer.elapsed();
  EXPECT_NEAR(elapsed_seconds, sleep_milliseconds/1e3, 1.5e-2);
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}
