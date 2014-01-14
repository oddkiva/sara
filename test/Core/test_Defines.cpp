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

//namespace DO { namespace Dev {
//
//  namespace ColorSpace {
//    struct RGB
//    { enum Channels { R = 0, G=1, B=2, NumChannels=3 }; };
//    struct RGBA
//    { enum Channels { R = 0, G=1, B=2, A=3, NumChannels=4 }; };
//    struct HSV
//    { enum Channels { H = 0, S=1, V=2, NumChannels=3 }; };
//    struct HSL
//    { enum Channels { H = 0, S=1, L=2, NumChannels=3 }; };
//    struct YUV 
//    { enum Channels { Y = 0, U=1, V=2, NumChannels=3 }; };
//    struct CMYK
//    { enum Channels { C = 0, M=1, Y=2, K=3, NumChannels=4 }; };
//    struct CIELab
//    { enum Channels { L = 0, a=1, b=2, NumChannels=3 }; };
//    struct CIEXyz
//    { enum Channels { X=0, Y=1, Z=2, NumChannels=3 }; };
//  }
//
//  namespace ColorLayout {
//    /*struct R {}; struct G {}; struct B {}; struct A {};
//    struct L {}; struct a {}; struct b {};
//    struct Y {}; struct U {}; struct V {};
//    struct X {}; struct Y {}; struct Z {};
//    struct C {}; struct M {}; struct Y {}; struct K {};*/
//    struct RGB { enum Order { C0=0, C1=1, C2=2 }; };
//    struct BGR { enum Order { C0=2, C1=1, C2=0 }; };
//    struct RGBA { enum Order { C0=0, C1=1, C2=2, C3=3 }; };
//    struct ARGB { enum Order { C0=1, C1=2, C2=3, C3=0 }; };
//    struct ABGR { enum Order { C0=3, C1=2, C2=1, C3=0 }; };
//  }
//
//  template <int N1, int N2, int N3>
//  struct PackedFormat { enum { Bits1=N1, Bits2=N2, Bits3=N3} };
//
//  template <int N1, int N2, int N3, int N4>
//  struct PackedFormat { enum { Bits1=N1, Bits2=N2, Bits3=N3, Bits4=N4 } };
//
//  template <typename T_, typename ColorSpace_, typename ColorLayout_>
//  class Pixel
//  {
//  public:
//    typedef T_ T;
//    typedef ColorSpace_ ColorSpace;
//    typedef ColorLayout_ ColorLayout;
//    typedef int ChannelType;
//    T val;
//
//    template<typename C>
//    int get(ChannelType channelValue);
//
//    template<typename C>
//    int set(ChannelType channelValue);
//  };
//
//} /* namespace Dev */
//} /* namespace DO */
//
//TEST(DO_Core_Test, pixelTest)
//{
//  using namespace DO::Dev;
//  ASSERT_EQ(sizeof(Pixel<char>), sizeof(char));
//  ASSERT_EQ(sizeof(Pixel<int>), sizeof(int));
//  ASSERT_EQ(sizeof(Pixel<Vector3d>), sizeof(Vector3d));
//}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}