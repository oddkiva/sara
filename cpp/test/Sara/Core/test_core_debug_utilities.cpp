// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/Debug Utilities"

#include <regex>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include "AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestDebugUtilities)

BOOST_AUTO_TEST_CASE(test_print_stage)
{
  stringstream buffer{};
  CoutRedirect cout_redirect{ buffer.rdbuf() };
  print_stage("Hello");
  auto text = buffer.str();

  BOOST_CHECK(text.find("Hello") != string::npos);
}

BOOST_AUTO_TEST_SUITE_END()
