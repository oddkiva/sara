// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <iostream>

#include <gtest/gtest.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.hpp"


using namespace std;
using namespace DO::Sara;


EventScheduler *global_scheduler;


class TestSleepFunctions: public testing::Test
{
protected:
  Window test_window_;

  TestSleepFunctions()
  {
    test_window_ = create_window(300, 300);
    global_scheduler->set_receiver(test_window_);
  }

  virtual ~TestSleepFunctions()
  {
    close_window(test_window_);
  }
};

TEST_F(TestSleepFunctions, test_millisleep)
{
  int delay_ms = 100;
  Timer timer;
  timer.restart();
  millisleep(delay_ms);
  double elapsed = timer.elapsedMs();

  double tol_ms = 10.;
  EXPECT_NEAR(elapsed, static_cast<double>(delay_ms), tol_ms);
}

TEST_F(TestSleepFunctions, test_microsleep)
{
  int delay_us = 100*1000; // 100 ms because 1000 us = 1 ms.
  Timer timer;
  timer.restart();
  microsleep(delay_us);
  double elapsed = timer.elapsedMs();

  double tol_us = 5;
  EXPECT_NEAR(elapsed, static_cast<double>(delay_us)*1e-3, tol_us);

}

int worker_thread(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

#undef main
int main(int argc, char **argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app_(argc, argv);

  // Create an event scheduler on the GUI thread.
  global_scheduler = new EventScheduler;
  // Connect the user thread and the event scheduler.
  QObject::connect(&get_user_thread(), SIGNAL(sendEvent(QEvent *, int)),
                   global_scheduler, SLOT(schedule_event(QEvent*, int)));

  // Run the worker thread
  gui_app_.register_user_main(worker_thread);
  int return_code = gui_app_.exec();

  // Cleanup and terminate
  delete global_scheduler;
  return return_code;
}