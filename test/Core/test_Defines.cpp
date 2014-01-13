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

#include "CoreTesting.hpp"

using namespace DO;
using namespace std;

TEST(DO_Core_Test, definesTest)
{
  cout << "DO++ version: " << DO_VERSION << endl;
  EXPECT_TRUE( !string(DO_VERSION).empty() );

  cout << "string source path: " << endl << srcPath("") << endl << endl;
  EXPECT_TRUE( string(srcPath("")).find("test/Core") != string::npos );
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}