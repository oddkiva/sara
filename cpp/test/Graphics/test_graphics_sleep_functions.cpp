// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2014-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE "Graphics/Sleep Functions"

#include <iostream>

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/GraphicsUtilities.hpp>

#include "event_scheduler.hpp"


using namespace std;
using namespace DO::Sara;


class TestFixtureForSleepFunctions
{
protected:
  Window test_window_;

public:
  TestFixtureForSleepFunctions()
  {
    test_window_ = create_window(300, 300);
  }

  ~TestFixtureForSleepFunctions()
  {
    close_window(test_window_);
  }
};


BOOST_FIXTURE_TEST_SUITE(TestSleepFunctions, TestFixtureForSleepFunctions)

BOOST_AUTO_TEST_CASE(test_millisleep)
{
  int delay_ms = 100;
  Timer timer;
  timer.restart();
  millisleep(delay_ms);
  double elapsed = timer.elapsed_ms();

  double tol_ms = 10.;
  BOOST_CHECK_SMALL(elapsed - static_cast<double>(delay_ms), tol_ms);
}

BOOST_AUTO_TEST_CASE(test_microsleep)
{
  int delay_us = 100*1000; // 100 ms because 1000 us = 1 ms.
  Timer timer;
  timer.restart();
  microsleep(delay_us);
  double elapsed = timer.elapsed_ms();

  double tol_us = 5;
  BOOST_CHECK_SMALL(elapsed - static_cast<double>(delay_us) * 1e-3, tol_us);
}

BOOST_AUTO_TEST_SUITE_END()

int worker_thread(int argc, char **argv)
{
  return boost::unit_test::unit_test_main([]() { return true; }, argc, argv);
}

int main(int argc, char **argv)
{
  // Create Qt Application.
  GraphicsApplication gui_app_(argc, argv);

  // Run the worker thread
  gui_app_.register_user_main(worker_thread);
  return gui_app_.exec();
}
