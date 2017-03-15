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

#define BOOST_TEST_MODULE "Core/Tensor Views and Conversions"

#include <vector>

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>


using namespace std;
using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestConversionImageToTensor)

BOOST_AUTO_TEST_CASE(test_grayscale_case)
{
  auto image = Image<float>{2, 3};
  image.matrix() << 0, 1, 2, 3, 4, 5;

  const auto tensor = to_cwh_tensor(image);

  auto true_tensor = Tensor<float, 2>{3, 2};
  true_tensor.matrix() << 0, 1, 2, 3, 4, 5;

  BOOST_CHECK_EQUAL(true_tensor.matrix(), tensor.matrix());
}

BOOST_AUTO_TEST_CASE(test_color_case)
{
  auto image = Image<Rgb32f>{2, 3};
  auto m = image.matrix();
  m.fill(Rgb32f{1., 2., 3.});

  m(0, 0) *= 0;
  m(0, 1) *= 1;
  m(1, 0) *= 2;
  m(1, 1) *= 3;
  m(2, 0) *= 4;
  m(2, 1) *= 5;

  const auto tensor = to_cwh_tensor(image);
  const auto r = tensor[0].matrix();
  const auto g = tensor[1].matrix();
  const auto b = tensor[2].matrix();

  auto true_r = Tensor<float, 2>{3, 2};
  auto true_g = Tensor<float, 2>{3, 2};
  auto true_b = Tensor<float, 2>{3, 2};
  true_r.matrix() << 0, 1, 2, 3, 4, 5;
  true_g.matrix() << 0, 2, 4, 6, 8, 10;
  true_b.matrix() << 0, 3, 6, 9, 12, 15;

  BOOST_CHECK_EQUAL(true_r.matrix(), r);
  BOOST_CHECK_EQUAL(true_g.matrix(), g);
  BOOST_CHECK_EQUAL(true_b.matrix(), b);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(TestTensorViews)

BOOST_AUTO_TEST_CASE(test_grayscale_case)
{
  auto image = Image<float>{2, 3};
  image.matrix() << 0, 1, 2, 3, 4, 5;

  const auto tensor = tensor_view(image);
  BOOST_CHECK_EQUAL(image.matrix(), tensor.matrix());
}

BOOST_AUTO_TEST_CASE(test_color_case)
{
  auto image = Image<Rgb32f>{2, 3};
  auto m = image.matrix();
  m.fill(Rgb32f{1., 2., 3.});

  m(0, 0) *= 0;
  m(0, 1) *= 1;
  m(1, 0) *= 2;
  m(1, 1) *= 3;
  m(2, 0) *= 4;
  m(2, 1) *= 5;

  const auto t = tensor_view(image);

  // Indexed by (y, x, c).
  BOOST_CHECK_EQUAL(Rgb32f::num_channels(), t.size(2));
  BOOST_CHECK_EQUAL(image.width(), t.size(1));
  BOOST_CHECK_EQUAL(image.height(), t.size(0));

  for (int y = 0; y < t.size(0); ++y)
    for (int x = 0; x < t.size(1); ++x)
      for (int c = 0; c < t.size(2); ++c)
        BOOST_REQUIRE_EQUAL((c + 1) * (x + y * t.size(1)), t(y, x, c));
}

BOOST_AUTO_TEST_CASE(test_matrix_case)
{
  constexpr auto W = 2, H = 3, M = 2, N = 3;
  auto img = Image<Matrix<float, M, N>>{W, H};  // Indexed by (y, x, j, i).

  auto img_elem = Matrix<float, M, N>{};
  std::iota(img_elem.data(), img_elem.data() + M * N, 0);
  img.flat_array().fill(img_elem);

  auto m = img.matrix();
  m(0, 0) *= 0;
  m(0, 1) *= 1;
  m(1, 0) *= 2;
  m(1, 1) *= 3;
  m(2, 0) *= 4;
  m(2, 1) *= 5;
  /*
   * [[0, 0, 0],  [[0, 2, 4],
   *  [0, 0, 0]]   [1, 3, 5]]
   *
   * [[0, 4, 8],  [[0, 6,12],
   *  [2, 6,10]]   [3, 9,15]]
   *
   * [[0, 8,16],  [[0,10,20],
   *  [4,12,20]]   [5,15,25]]
   *
   */

  const auto t = tensor_view(img);  // Indexed by (y, x, j, i).

  BOOST_CHECK_EQUAL(M, t.size(3));
  BOOST_CHECK_EQUAL(N, t.size(2));
  BOOST_CHECK_EQUAL(W, t.size(1));
  BOOST_CHECK_EQUAL(H, t.size(0));
  BOOST_CHECK_EQUAL(img.width(), t.size(1));
  BOOST_CHECK_EQUAL(img.height(), t.size(0));

  for (int y = 0; y < t.size(0); ++y)
    for (int x = 0; x < t.size(1); ++x)
      for (int j = 0; j < t.size(2); ++j)
        for (int i = 0; i < t.size(3); ++i)
          BOOST_REQUIRE_EQUAL(img_elem(i, j) * (y * W + x), t(Vector4i{y, x, j, i}));
}

BOOST_AUTO_TEST_CASE(test_array_case)
{
  constexpr auto W = 2, H = 3, M = 2, N = 3;
  auto img = Image<Array<float, M, N>>{W, H};  // Indexed by (y, x, j, i).

  auto img_elem = Array<float, M, N>{};
  std::iota(img_elem.data(), img_elem.data() + M * N, 0);
  img.flat_array().fill(img_elem);

  auto m = img.matrix();
  m(0, 0) *= 0;
  m(0, 1) *= 1;
  m(1, 0) *= 2;
  m(1, 1) *= 3;
  m(2, 0) *= 4;
  m(2, 1) *= 5;
  /*
   * [[0, 0, 0],  [[0, 2, 4],
   *  [0, 0, 0]]   [1, 3, 5]]
   *
   * [[0, 4, 8],  [[0, 6,12],
   *  [2, 6,10]]   [3, 9,15]]
   *
   * [[0, 8,16],  [[0,10,20],
   *  [4,12,20]]   [5,15,25]]
   *
   */

  const auto t = tensor_view(img);  // Indexed by (y, x, j, i).

  BOOST_CHECK_EQUAL(M, t.size(3));
  BOOST_CHECK_EQUAL(N, t.size(2));
  BOOST_CHECK_EQUAL(W, t.size(1));
  BOOST_CHECK_EQUAL(H, t.size(0));
  BOOST_CHECK_EQUAL(img.width(), t.size(1));
  BOOST_CHECK_EQUAL(img.height(), t.size(0));

  for (int y = 0; y < t.size(0); ++y)
    for (int x = 0; x < t.size(1); ++x)
      for (int j = 0; j < t.size(2); ++j)
        for (int i = 0; i < t.size(3); ++i)
          BOOST_CHECK_EQUAL(img_elem(i, j) * (y * W + x),
                            t(Vector4i{y, x, j, i}));
}

BOOST_AUTO_TEST_SUITE_END()
