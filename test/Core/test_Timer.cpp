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
#include <TinyThread++/source/tinythread.h>

using namespace DO;
using namespace std;

inline void wait(unsigned milliseconds)
{
#ifdef _WIN32
  Sleep(milliseconds);
#else
  usleep(milliseconds*1e3);
#endif
}

// Thread function: Detach
void oneSecondSleep(void *)
{
  // We don't do anything much, just sleep a little...
  tthread::this_thread::sleep_for(tthread::chrono::milliseconds(1000));
}

TEST(DO_Core_Test,  testTimer)
{
  Timer timer;
  HighResTimer hrTimer;
  double elapsedTimeMs;
  double elapsedTimeS;
  double sleepTimeMs = 1e3;

  hrTimer.restart();
  wait(sleepTimeMs);
  elapsedTimeMs =  hrTimer.elapsedMs();
  EXPECT_NEAR(elapsedTimeMs, sleepTimeMs, 100);

  hrTimer.restart();
  wait(sleepTimeMs);
  elapsedTimeS = hrTimer.elapsed();
  EXPECT_NEAR(elapsedTimeS, sleepTimeMs/1e3, 1e-3);
  
  timer.restart();
  // Start the child thread
  tthread::thread t(oneSecondSleep, 0);
  // Wait for the thread to finish
  t.join();
  elapsedTimeS = timer.elapsed();
  EXPECT_NEAR(elapsedTimeS, sleepTimeMs/1e3, 1e-3);
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}