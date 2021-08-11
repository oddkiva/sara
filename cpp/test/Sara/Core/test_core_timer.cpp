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

#define BOOST_TEST_MODULE "Core/Timer Class"

#include <chrono>
#include <thread>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>


using namespace DO::Sara;
using namespace std;

inline void millisleep(unsigned milliseconds)
{
  chrono::milliseconds duration(milliseconds);
  this_thread::sleep_for(duration);
}

BOOST_AUTO_TEST_SUITE(TestTimer)

BOOST_AUTO_TEST_CASE(test_timer)
{
  Timer timer{};
  auto elapsed_milliseconds = double{};
  auto elapsed_seconds = double{};
  constexpr auto sleep_milliseconds = unsigned{ 580 };

  timer.restart();
  millisleep(sleep_milliseconds);
  elapsed_milliseconds =  timer.elapsed_ms();
  BOOST_CHECK_SMALL(elapsed_milliseconds - sleep_milliseconds, 100.);

  timer.restart();
  millisleep(sleep_milliseconds);
  elapsed_seconds = timer.elapsed();
  BOOST_CHECK_SMALL(elapsed_seconds - sleep_milliseconds / 1e3, 5e-2);
}

BOOST_AUTO_TEST_CASE(test_tictoc)
{
  constexpr auto sleep_milliseconds = unsigned{ 580 };

  tic();
  millisleep(sleep_milliseconds);
  toc("sleep time");

  const auto elapsed_milliseconds = TicToc::instance().elapsed_ms();
  BOOST_CHECK_SMALL(elapsed_milliseconds - sleep_milliseconds, 100.);
}

BOOST_AUTO_TEST_SUITE_END()
