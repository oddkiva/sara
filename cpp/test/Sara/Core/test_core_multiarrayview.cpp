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

#define BOOST_TEST_MODULE "Core/MultiArray/MultiArrayView Class"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>
#include <DO/Sara/Core/MultiArray/DataTransformations.hpp>
#include <DO/Sara/Core/MultiArray/Slice.hpp>
#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/Core/Tensor.hpp>


using namespace std;
using namespace DO::Sara;


template <typename T, int N>
auto as_col_vector_list(const TensorView_<T, 2>& t)
{
  using Vector = Eigen::Matrix<T, N, 1>;
  return TensorView_<Vector, 1>{
      const_cast<Vector*>(reinterpret_cast<const Vector*>(t.data())),
      t.size(0)};
}

template <typename T, int M, int N>
auto as_tensor(const Matrix<T, M, N>& x)
{
  return TensorView_<T, 1>{const_cast<T*>(x.data()), x.rows(), x.cols()};
}

BOOST_AUTO_TEST_SUITE(TestMultiArrayView)

BOOST_AUTO_TEST_CASE(test_multiarrayview_empty)
{
  {
    auto array_view = MultiArrayView<int, 2>{};
    BOOST_CHECK(array_view.empty());
  }

  {
    auto r = std::vector<int>(3 * 4);
    auto array_view = MultiArrayView<int, 2>{r.data(), {3, 4}};
    BOOST_CHECK(!array_view.empty());
  }
}

BOOST_AUTO_TEST_CASE(test_multiarrayview_1d)
{
  auto r = std::vector<int>(10);
  std::iota(std::begin(r), std::end(r), 0);

  const auto const_r_view = MultiArrayView<int, 1>{r.data(), int(r.size())};
  for (auto i = 0; i < int(const_r_view.size()); ++i)
    BOOST_CHECK_EQUAL(const_r_view(i), i);

  auto r_view = MultiArrayView<int, 1>{r.data(), int(r.size())};
  for (auto i = 0u; i < const_r_view.size(); ++i)
    r_view(i) = 0;

  for (auto i = 0u; i < const_r_view.size(); ++i)
    BOOST_CHECK_EQUAL(r_view(i), 0);
}

BOOST_AUTO_TEST_CASE(test_multiarrayview_cast)
{
  auto r = std::vector<int>(10);
  std::iota(std::begin(r), std::end(r), 0);

  const auto const_r_view = tensor_view(r);
  for (auto i = 0; i < int(const_r_view.size()); ++i)
    BOOST_CHECK_EQUAL(const_r_view(i), i);

  auto rd = const_r_view.cast<double>();

  auto true_rd = Tensor_<double, 1>(10);
  std::iota(std::begin(true_rd), std::end(true_rd), 0);

  BOOST_CHECK(true_rd.vector() == rd.vector());
}

BOOST_AUTO_TEST_CASE(test_multiarrayview_flatten)
{
  auto r = range(10).reshape(Vector2i{2, 5});

  auto r_flatten = r.flatten();

  BOOST_CHECK(r_flatten == range(10));
}

BOOST_AUTO_TEST_CASE(test_multiarrayview_crop)
{
  const auto X = range(7 * 7).cast<float>().reshape(Vector2i{7, 7});
  std::cout << X.matrix() << std::endl;

  const auto X_cropped = X.cropped_view({1, 1}, {6, 6}, {2, 2});
  std::cout << X_cropped.sizes() << std::endl;
  for (auto i = 0; i < X_cropped.size(0); ++i)
  {
    for (auto j = 0; j < X_cropped.size(1); ++j)
      std::cout << X_cropped(i, j) << " ";
    std::cout << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(test_slice_view)
{
  const auto X = range(24).cast<float>().reshape(Vector2i{6, 4});
  SARA_DEBUG << "X =\n" << X.matrix() << std::endl;

  auto X_sliced = slice(X, {{1, 6, 2}, {1, 4, 2}}).make_copy();
  SARA_DEBUG << "X_sliced =\n" << X_sliced.matrix() << std::endl;

  static_assert(int(decltype(X_sliced)::StorageOrder) == int(Eigen::RowMajor));

  for (int y = 0; y < X_sliced.size(0); ++y)
    for (int x = 0; x < X_sliced.size(1); ++x)
      SARA_DEBUG << format("X_sliced(%d, %d) = %f", y, x,
                           X_sliced(Vector2i{y, x}))
                 << std::endl;
}

BOOST_AUTO_TEST_CASE(test_filtered_transformed_operations)
{
  auto v = arange(-10.f, 10.f, 1.f);

  auto v2 = v
    | filtered([](float x) { return x > 0; })
    | transformed([](float x) { return int(x * x); });

  SARA_DEBUG << v2.row_vector() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_filtered_transformed_usage_case)
{
  auto indices = range(10);

  auto cheiral = Eigen::Array<int, 1, Eigen::Dynamic>(10);
  cheiral << true, true, true, false, false, true, true, false, false, false;

  auto inliers = Eigen::Array<bool, 1, Eigen::Dynamic>(10);
  inliers << false, true, false, false, false, true, true, false, true, false;

  auto P = Matrix34d{};
  P.leftCols(3) = Matrix3d::Identity() * 2;
  P.col(3).setZero();
  SARA_CHECK(P);

  auto X = MatrixXd{4, 10};
  for (int i = 0; i < X.cols(); ++i)
    X.col(i) = Vector4d::Ones() * i;

  SARA_DEBUG << "X =\n" << X.matrix() << std::endl;

  MatrixXd PX = P * X.matrix();
  SARA_DEBUG << "PX =\n" << PX << std::endl;

  auto indices_filtered =
      indices | filtered([&](int i) { return cheiral(i) && inliers(i); });
}

BOOST_AUTO_TEST_SUITE_END()
