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

#define BOOST_TEST_MODULE "ImageProcessing/Color Statistics"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/ColorStatistics.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestColorStatistics)

BOOST_AUTO_TEST_CASE(test_color_mean)
{
  auto images = vector<Image<Rgb64f>>{
    Image<Rgb64f>{2, 2},
    Image<Rgb64f>{2, 2}
  };

  images[0].flat_array().fill(Rgb64f::Zero());
  images[1].flat_array().fill(Rgb64f::Ones());

  const auto m = color_sample_mean_vector(images.begin(), images.end());
  BOOST_CHECK_EQUAL(m, (Vector3d::Ones() * 0.5).eval());

  const auto m1 = online_color_covariance(images.begin(), images.end()).first;
  BOOST_CHECK_EQUAL(m1, (Vector3d::Ones() * 0.5).eval());
}

BOOST_AUTO_TEST_CASE(test_color_covariance_matrix)
{
  auto images = vector<Image<Rgb64f>>{
    Image<Rgb64f>{2, 2},
    Image<Rgb64f>{2, 2}
  };

  images[0].flat_array().fill(Rgb64f::Zero());
  images[1].flat_array().fill(Rgb64f::Ones());


  const auto m = color_sample_mean_vector(images.begin(), images.end());
  const auto cov = color_sample_covariance_matrix(images.begin(), images.end(), m);

  const auto true_cov = (Matrix3d::Ones() * 0.25 * 8 / 7).eval();
  BOOST_CHECK_EQUAL(true_cov, cov);

  const auto stats = online_color_covariance(images.begin(), images.end());
  BOOST_CHECK_EQUAL(true_cov, stats.second);
}

BOOST_AUTO_TEST_CASE(test_color_pca)
{
  auto images = vector<Image<Rgb64f>>{
    Image<Rgb64f>{2, 2},
    Image<Rgb64f>{2, 2}
  };

  images[0].flat_array().fill(Rgb64f::Zero());
  images[1].flat_array().fill(Rgb64f::Ones());


  const auto m = color_sample_mean_vector(images.begin(), images.end());
  const auto cov = color_sample_covariance_matrix(images.begin(), images.end(), m);

  auto pca = color_pca(cov);
  const auto& U = pca.first;
  const auto& S = pca.second;

  BOOST_CHECK_LE(S.tail(2).norm(), 1e-8);
  BOOST_CHECK_LE(abs(U.col(0).dot(U.col(1))), 1e-8);
  BOOST_CHECK_LE(abs(U.col(0).dot(U.col(2))), 1e-8);
  BOOST_CHECK_LE(abs(U.col(1).dot(U.col(2))), 1e-8);

  for (int i = 0; i < 3; ++i)
    BOOST_CHECK_LE(abs(U.col(i).norm() - 1), 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
