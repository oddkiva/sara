// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Shakti/CUDA/Utilities"

#include <DO/Shakti/Cuda/Utilities.hpp>

#include <boost/test/unit_test.hpp>


namespace sc = DO::Shakti;

BOOST_AUTO_TEST_CASE(test_check_cuda_devices)
{
  const auto cuda_devices = sc::get_devices();
  std::cout << cuda_devices.front() << std::endl;
  BOOST_CHECK(!cuda_devices.empty());
}

BOOST_AUTO_TEST_CASE(test_string_format)
{
  const auto hello = sc::format("hello %d", 0);
  SHAKTI_STDOUT << hello << std::endl;
  BOOST_CHECK_EQUAL(hello, "hello 0");
}


BOOST_AUTO_TEST_CASE(test_timer)
{
  auto timer = sc::Timer{};
  timer.restart();
  const auto elapsed = timer.elapsed_ms();
  SHAKTI_STDOUT << elapsed << " ms" << std::endl;
  BOOST_CHECK_GT(elapsed, 0);
}