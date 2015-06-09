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

#include <exception>

#include <gtest/gtest.h>

#include <DO/Sara/ImageProcessing/GaussianPyramid.hpp>


using namespace std;
using namespace DO::Sara;


template <class ChannelType>
class TestGaussianPyramid : public testing::Test {};

typedef testing::Types<float, double, Rgb32f, Rgb64f> ChannelTypes;

TYPED_TEST_CASE_P(TestGaussianPyramid);

TYPED_TEST_P(TestGaussianPyramid, test_gaussian_pyramid)
{
  typedef TypeParam T;
  Image<T> I(16, 16);
  I.matrix().fill(PixelTraits<T>::max());

  ImagePyramid<T> G(gaussian_pyramid(I, ImagePyramidParams(-1)));

  ImagePyramid<T> D(difference_of_gaussians_pyramid(G));

  ImagePyramid<T> L(laplacian_pyramid(G));
}

REGISTER_TYPED_TEST_CASE_P(TestGaussianPyramid, test_gaussian_pyramid);
INSTANTIATE_TYPED_TEST_CASE_P(DO_SARA_ImageProcessing_Pyramid_Test,
                              TestGaussianPyramid, ChannelTypes);

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}