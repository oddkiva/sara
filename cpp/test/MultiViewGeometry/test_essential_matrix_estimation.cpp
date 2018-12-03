// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#define BOOST_TEST_MODULE "MultiViewGeometry/Essential Matrix"

#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <boost/test/unit_test.hpp>

#include <iomanip>
#include <sstream>


using namespace std;
using namespace DO::Sara;


template <typename T>
void print_3d_array(const TensorView_<T, 3>& x)
{
  const auto max = x.flat_array().abs().maxCoeff();
  std::stringstream ss;
  ss << max;
  const auto pad_size = ss.str().size();


  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << std::setw(pad_size) << x(i,j,k);
        if (k != x.size(2) - 1)
          cout << ", ";
      }
      cout << "]";

      if (j != x.size(1) - 1)
        cout << ", ";
      else
        cout << "]";
    }

    if (i != x.size(0) - 1)
      cout << ",\n ";
  }
  cout << "]" << endl;
}

void print_3d_array(const TensorView_<float, 3>& x)
{
  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << fixed << x(i,j,k);
        if (k != x.size(2) - 1)
          cout << ", ";
      }
      cout << "]";

      if (j != x.size(1) - 1)
        cout << ", ";
      else
        cout << "]";
    }

    if (i != x.size(0) - 1)
      cout << ",\n ";
  }
  cout << "]" << endl;
}


BOOST_AUTO_TEST_SUITE(TestMultiViewGeometry)

BOOST_AUTO_TEST_CASE(test_range)
{
  auto a = range(3);
  BOOST_CHECK(a.vector() == Vector3i(0, 1, 2));
}

BOOST_AUTO_TEST_CASE(test_random_shuffle)
{
  auto a = range(4);
  a = shuffle(a);
  BOOST_CHECK(a.vector() != Vector4i(0, 1, 2, 3));
}

BOOST_AUTO_TEST_CASE(test_random_samples)
{
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 5;
  constexpr auto num_data_points = 10;
  auto samples = random_samples(num_samples, sample_size, num_data_points);

  BOOST_CHECK_EQUAL(samples.size(0), num_samples);
  BOOST_CHECK_EQUAL(samples.size(1), sample_size);
  BOOST_CHECK(samples.matrix().minCoeff() >=  0);
  BOOST_CHECK(samples.matrix().maxCoeff() <  10);
}


BOOST_AUTO_TEST_CASE(test_extract_centers)
{
  auto features = std::vector<OERegion>{{Point2f::Ones() * 0, 1.f},
                                        {Point2f::Ones() * 1, 1.f},
                                        {Point2f::Ones() * 2, 1.f}};

  const auto x = extract_centers(features);
  auto expected_x = Tensor_<float, 2>{x.sizes()};
  expected_x.matrix() <<
    0, 0,
    1, 1,
    2, 2;

  BOOST_CHECK(x.matrix() == expected_x.matrix());

  const auto X = homogeneous(x);
  auto expected_X = Tensor_<float, 2>{X.sizes()};
  expected_X.matrix() <<
    0, 0, 1,
    1, 1, 1,
    2, 2, 1;
  BOOST_CHECK(X.matrix() == expected_X.matrix());
}

BOOST_AUTO_TEST_CASE(test_to_point_indices)
{
  constexpr auto num_matches = 5;
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 4;

  auto matches = Tensor_<int, 2>{num_matches, 2};
  matches.matrix() <<
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 0;

  auto samples = Tensor_<int, 2>{num_samples, sample_size};
  samples.matrix() <<
    0, 1, 2, 3,
    4, 2, 3, 1;

  auto point_indices = to_point_indices(samples, matches);

  auto expected_point_indices = Tensor_<int, 3>{num_samples, sample_size, 2};
  expected_point_indices[0].matrix() <<
    0, 0,
    1, 1,
    2, 2,
    3, 3;
  expected_point_indices[1].matrix() <<
    4, 0,
    2, 2,
    3, 3,
    1, 1;

  BOOST_CHECK(point_indices.vector() == expected_point_indices.vector());
}

BOOST_AUTO_TEST_CASE(test_to_coordinates)
{
  constexpr auto num_matches = 5;
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 4;

  const auto features1 = std::vector<OERegion>{{Point2f::Ones() * 0, 1.f},
                                               {Point2f::Ones() * 1, 1.f},
                                               {Point2f::Ones() * 2, 1.f}};
  const auto features2 = std::vector<OERegion>{{Point2f::Ones() * 1, 1.f},
                                               {Point2f::Ones() * 2, 1.f},
                                               {Point2f::Ones() * 3, 1.f}};

  const auto points1 = extract_centers(features1);
  const auto points2 = extract_centers(features2);

  auto matches = Tensor_<int, 2>{num_matches, 2};
  matches.matrix() <<
    0, 0,
    1, 1,
    2, 2,
    0, 1,
    1, 2;

  auto samples = Tensor_<int, 2>{num_samples, sample_size};
  samples.matrix() <<
    0, 1, 2, 3,
    1, 2, 3, 4;

  const auto point_indices = to_point_indices(samples, matches);
  const auto coords = to_coordinates(point_indices, points1, points2);

  //                                        N            K            P  C
  auto expected_coords = Tensor_<float, 4>{{num_samples, sample_size, 2, 2}};
  expected_coords[0].flat_array() <<
    0.f, 0.f, 1.f, 1.f,
    1.f, 1.f, 2.f, 2.f,
    2.f, 2.f, 3.f, 3.f,
    0.f, 0.f, 2.f, 2.f;

  expected_coords[1].flat_array() <<
    1.f, 1.f, 2.f, 2.f,
    2.f, 2.f, 3.f, 3.f,
    0.f, 0.f, 2.f, 2.f,
    1.f, 1.f, 3.f, 3.f;

  BOOST_CHECK(expected_coords.vector() == coords.vector());
  BOOST_CHECK(expected_coords.sizes() == coords.sizes());


  const auto coords_t = coords.transpose({0, 2, 1, 3});
  const auto sample1 = coords_t[0];

  auto expected_sample1 = Tensor_<float, 3>{sample1.sizes()};
  expected_sample1.flat_array() <<
    // P1
    0.f, 0.f,
    1.f, 1.f,
    2.f, 2.f,
    0.f, 0.f,
    // P2
    1.f, 1.f,
    2.f, 2.f,
    3.f, 3.f,
    2.f, 2.f;

  //print_3d_array(expected_sample1);
  //print_3d_array(sample1);
  BOOST_CHECK(expected_sample1.vector() == sample1.vector());
}


BOOST_AUTO_TEST_CASE(test_compute_normalizer)
{
  auto X = Tensor_<float, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  auto T = compute_normalizer(X);

  Matrix3f expected_T;
  expected_T <<
    0.5, 0.0, -0.5,
    0.0, 0.5, -0.5,
    0.0, 0.0,  1.0;

  BOOST_CHECK((T - expected_T).norm() < 1e-12);
}

BOOST_AUTO_TEST_CASE(test_apply_transform)
{
  auto X = Tensor_<float, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  // From Oxford affine covariant features dataset, graf, H1to5P homography.
  auto H = Matrix3f{};
  H <<
    6.2544644e-01,  5.7759174e-02,  2.2201217e+02,
    2.2240536e-01,  1.1652147e+00, -2.5605611e+01,
    4.9212545e-04, -3.6542424e-05,  1.0000000e+00;

  auto HX = apply_transform(H, X);

  BOOST_CHECK(HX.matrix().col(2) == Vector3f::Ones());
}


BOOST_AUTO_TEST_CASE(test_skew_symmetric_matrix)
{
  Vector3f t{1, 2, 3};
  Matrix3f T;
  T <<
     0, -3,  2,
     3,  0, -1,
    -2,  1,  0;

  BOOST_CHECK(skew_symmetric_matrix(t) == T);
}

BOOST_AUTO_TEST_SUITE_END()
