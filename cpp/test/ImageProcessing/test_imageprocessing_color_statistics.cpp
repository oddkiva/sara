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

#include <gtest/gtest.h>

#include <DO/Sara/ImageProcessing/ColorStatistics.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestColorStatistics, test_color_mean)
{
  auto images = vector<Image<Rgb64f>>{
    Image<Rgb64f>{2, 2},
    Image<Rgb64f>{2, 2}
  };

  images[0].array().fill(Rgb64f::Zero());
  images[1].array().fill(Rgb64f::Ones());

  const auto m = color_sample_mean_vector(images.begin(), images.end());
  EXPECT_MATRIX_EQ(m, (Vector3d::Ones() * 0.5).eval());
}

TEST(TestColorStatistics, test_color_covariance_matrix)
{
  auto images = vector<Image<Rgb64f>>{
    Image<Rgb64f>{2, 2},
    Image<Rgb64f>{2, 2}
  };

  images[0].array().fill(Rgb64f::Zero());
  images[1].array().fill(Rgb64f::Ones());


  const auto m = color_sample_mean_vector(images.begin(), images.end());
  const auto cov = color_sample_covariance_matrix(images.begin(), images.end(), m);

  const auto true_cov = (Matrix3d::Ones() * 0.25 * 8 / 7).eval();
  EXPECT_MATRIX_EQ(true_cov, cov);
}

TEST(TestColorStatistics, test_color_pca)
{
  auto images = vector<Image<Rgb64f>>{
    Image<Rgb64f>{2, 2},
    Image<Rgb64f>{2, 2}
  };

  images[0].array().fill(Rgb64f::Zero());
  images[1].array().fill(Rgb64f::Ones());


  const auto m = color_sample_mean_vector(images.begin(), images.end());
  const auto cov = color_sample_covariance_matrix(images.begin(), images.end(), m);

  auto pca = color_pca(cov);
  const auto& U = pca.first;
  const auto& S = pca.second;

  EXPECT_LE(S.tail(2).norm(), 1e-8);
  EXPECT_LE(abs(U.col(0).dot(U.col(1))), 1e-8);
  EXPECT_LE(abs(U.col(0).dot(U.col(2))), 1e-8);
  EXPECT_LE(abs(U.col(1).dot(U.col(2))), 1e-8);

  for (int i = 0; i < 3; ++i)
    EXPECT_LE(abs(U.col(i).norm() - 1), 1e-8);
}


int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
