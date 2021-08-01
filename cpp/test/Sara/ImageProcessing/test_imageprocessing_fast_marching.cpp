// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "ImageProcessing/Level Sets/Fast Marching Method"

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/FastMarching.hpp>

#include "../AssertHelpers.hpp"

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_CASE(test_min_coeff_in_fast_marching)
{
  constexpr auto N = 3;
  auto us = Eigen::Matrix<float, N, 1>{};
  us << 0, 1, 2;

  // Implementation details check.
  auto umins = Eigen::Matrix<float, N - 1, N>{};
  for (auto j = 0; j < N; ++j)
  {
    if (j == 0)
      umins.col(j) << us.segment(1, N - 1);
    else if (j == N - 1)
      umins.col(j) << us.head(N - 1);
    else
      umins.col(j) << us.head(j), us.segment(j + 1, N - j - 1);
  }

  auto umins_true = Eigen::Matrix<float, N - 1, N>{};
  umins_true << 1, 0, 0,
                2, 2, 1;

  BOOST_CHECK(umins == umins_true);

  // Check the helper function.
  const auto min_coeff = sara::FastMarching<float, 3>::find_min_coefficient(us);
  BOOST_CHECK_EQUAL(min_coeff, 0.f);
}

BOOST_AUTO_TEST_CASE(test_fast_marching_2d)
{
  auto displacements = sara::Image<float>{10, 10};
  displacements.flat_array().fill(1);
  auto fm = sara::FastMarching{displacements};
  fm.initialize_alive_points({Eigen::Vector2i{5, 5}});
  fm.run();

  // Check that we have a radial propagation.
  auto distances = sara::Image<float>{10, 10};
  for (auto y = 0; y < distances.height(); ++y)
    for (auto x = 0; x < distances.width(); ++x)
      distances(x, y) = (Eigen::Vector2f(x, y) - Eigen::Vector2f(5, 5)).norm();

  const Eigen::MatrixXf d = fm._distances.matrix().block(  //
      fm._margin.y(),                                      //
      fm._margin.x(),                                      //
      fm._distances.height() - 2 * fm._margin.y(),         //
      fm._distances.width() - 2 * fm._margin.x()           //
  );
  const Eigen::MatrixXf d_true = distances.matrix().block(  //
      fm._margin.y(),                                       //
      fm._margin.x(),                                       //
      fm._distances.height() - 2 * fm._margin.y(),          //
      fm._distances.width() - 2 * fm._margin.x()            //
  );

  // std::cout << "CALCULATED =" << std::endl;
  // std::cout << d << std::endl;
  // std::cout << "EXPECTED =" << std::endl;
  // std::cout << d_true << std::endl;

  BOOST_CHECK_LE((d - d_true).cwiseAbs().maxCoeff(), 0.5f);
}

BOOST_AUTO_TEST_CASE(test_fast_marching_3d)
{
  namespace sara = DO::Sara;
  auto displacements = sara::Image<float, 3>(10, 10, 10);
  displacements.flat_array().fill(1);

  auto fm = sara::FastMarching{displacements};

  // Enumerate by hand all neighboring points except the zero vector.
  const auto deltas_true = std::array<Eigen::Vector3i, 26>{
      //               x   y   z
      Eigen::Vector3i{-1, -1, -1},  //
      Eigen::Vector3i{ 0, -1, -1},  //
      Eigen::Vector3i{ 1, -1, -1},  //

      Eigen::Vector3i{-1,  0, -1},  //
      Eigen::Vector3i{ 0,  0, -1},  //
      Eigen::Vector3i{ 1,  0, -1},  //

      Eigen::Vector3i{-1,  1, -1},  //
      Eigen::Vector3i{ 0,  1, -1},  //
      Eigen::Vector3i{ 1,  1, -1},  //

      Eigen::Vector3i{-1, -1,  0},  //
      Eigen::Vector3i{ 0, -1,  0},  //
      Eigen::Vector3i{ 1, -1,  0},  //

      Eigen::Vector3i{-1,  0,  0},  //
      Eigen::Vector3i{ 1,  0,  0},  //

      Eigen::Vector3i{-1,  1,  0},  //
      Eigen::Vector3i{ 0,  1,  0},  //
      Eigen::Vector3i{ 1,  1,  0},  //

      Eigen::Vector3i{-1, -1,  1},  //
      Eigen::Vector3i{ 0, -1,  1},  //
      Eigen::Vector3i{ 1, -1,  1},  //

      Eigen::Vector3i{-1,  0,  1},  //
      Eigen::Vector3i{ 0,  0,  1},  //
      Eigen::Vector3i{ 1,  0,  1},  //

      Eigen::Vector3i{-1,  1,  1},  //
      Eigen::Vector3i{ 0,  1,  1},  //
      Eigen::Vector3i{ 1,  1,  1}   //
  };

  BOOST_CHECK_EQUAL(fm._deltas.size(), 26);
  BOOST_CHECK(fm._deltas == deltas_true);
  fm.initialize_alive_points({Eigen::Vector3i{5, 5, 5}});
  fm.run();

  const auto& distances = fm._distances;
  const auto& margin = fm._margin;

  for (auto z = margin.z(); z < distances.depth() - margin.z(); ++z)
  {
    // std::cout << "Plane z = " << z << std::endl
    //           << sara::tensor_view(distances)[z].matrix().block(1, 1, 8, 8)
    //           << std::endl;
    for (auto y = margin.y(); y < distances.height() - margin.y(); ++y)
    {
      for (auto x = margin.x(); x < distances.width() - margin.x(); ++x)
      {
        const auto d_xyz =
            (Eigen::Vector3i(x, y, z).cast<float>() - Eigen::Vector3f(5, 5, 5)).norm();
        BOOST_REQUIRE_LE(std::abs(distances(x, y, z) - d_xyz), 0.8f);
      }
    }
  }
}
