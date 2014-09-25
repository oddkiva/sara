// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2014 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Core/Image/Image.hpp>


using namespace std;
using namespace DO;


// ========================================================================== //
// Run the tests.
int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}