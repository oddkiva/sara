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

#define BOOST_TEST_MODULE "Core/Numpy-like Functions"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Numpy.hpp>

#include <boost/test/unit_test.hpp>


using namespace DO::Sara;
namespace eigen = DO::Sara::EigenExt;


BOOST_AUTO_TEST_CASE(test_range)
{
  auto a = range(3);
  BOOST_CHECK(a.vector() == Vector3i(0, 1, 2));
}

BOOST_AUTO_TEST_CASE(test_arange)
{
  static_assert(std::is_same_v<decltype(double{} * float{} * int{}), double>);

  {
    const auto samples = eigen::arange(0.5, 1.5, 0.1);
    auto true_samples = Eigen::VectorXd(10);
    true_samples << 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4;
    BOOST_CHECK_LE((samples - true_samples).norm() / true_samples.norm(),
                   1e-12);
  }

  {
    const auto samples = eigen::arange(0., 1., 0.1);
    auto true_samples = Eigen::VectorXd(10);
    true_samples << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
    BOOST_CHECK_LE((samples - true_samples).norm() / true_samples.norm(),
                   1e-12);
  }

  {
    // That's the precision limit here for clang.
    const auto samples = eigen::arange(0., 1.000000000000001, 0.1);
    auto true_samples = Eigen::VectorXd(11);
    true_samples << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0;
    BOOST_CHECK_LE((samples - true_samples).norm() / true_samples.norm(),
                   1e-12);
  }
}

BOOST_AUTO_TEST_CASE(test_arange_2)
{
  {
    const auto samples = arange(0.5, 1.5, 0.1);
    auto true_samples = Eigen::VectorXd(10);
    true_samples << 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4;
    BOOST_CHECK_LE(
        (samples.vector() - true_samples).norm() / true_samples.norm(), 1e-12);
  }

  {
    const auto samples = arange(0., 1., 0.1);
    auto true_samples = Eigen::VectorXd(10);
    true_samples << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9;
    BOOST_CHECK_LE((samples.vector() - true_samples).norm() / true_samples.norm(),
                   1e-12);
  }

  {
    // That's the precision limit here for clang.
    const auto samples = arange(0., 1.000000000000001, 0.1);
    auto true_samples = Eigen::VectorXd(11);
    true_samples << 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0;
    BOOST_CHECK_LE(
        (samples.vector() - true_samples).norm() / true_samples.norm(), 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(test_vstack)
{
  auto matrices = std::vector<Eigen::MatrixXi>{
    Eigen::MatrixXi::Ones(1, 3),
    Eigen::MatrixXi::Ones(3, 3) * 2,
    Eigen::MatrixXi::Ones(5, 3) * 3,
  };

  auto vstack_res = eigen::vstack(matrices);

  auto true_vstack_res = Eigen::MatrixXi(9, 3);
  true_vstack_res <<                   //
     Eigen::MatrixXi::Ones(1, 3),      //
     Eigen::MatrixXi::Ones(3, 3) * 2,  //
     Eigen::MatrixXi::Ones(5, 3) * 3;

  BOOST_CHECK(vstack_res == true_vstack_res);
}

BOOST_AUTO_TEST_CASE(test_vstack_tensors)
{
  auto t0 = Tensor<int, 2>{{1, 3}};
  auto t1 = Tensor<int, 2>{{3, 3}};
  t0.matrix() = Eigen::MatrixXi::Ones(1, 3);
  t1.matrix() = Eigen::MatrixXi::Ones(3, 3) * 2;

  auto vstack_res = vstack(t0, t1);

  auto vstack_true = Eigen::MatrixXi(4, 3);
  vstack_true <<                        //
      Eigen::MatrixXi::Ones(1, 3),      //
      Eigen::MatrixXi::Ones(3, 3) * 2;  //

  SARA_DEBUG << "vstack =\n" << vstack_res.matrix() << std::endl;
  BOOST_CHECK(vstack_res.matrix() == vstack_true);
}

BOOST_AUTO_TEST_CASE(test_meshgrid)
{
   auto x = Eigen::MatrixXd(3, 1);
   auto y = Eigen::MatrixXd(2, 1);

   x << 0., 0.5, 1.;
   y << 0., 1.;

   Eigen::MatrixXd xv, yv;
   std::tie(xv, yv) = eigen::meshgrid(x, y);

   auto true_xv = Eigen::MatrixXd(3, 2);
   auto true_yv = Eigen::MatrixXd(3, 2);
   true_xv << 0. , 0  ,
              0.5, 0.5,
              1. , 1. ;
   true_yv << 0., 1.,
              0., 1.,
              0., 1.;

   BOOST_CHECK_EQUAL(xv.rows(), 3);
   BOOST_CHECK_EQUAL(xv.cols(), 2);
   BOOST_CHECK(xv == true_xv);

   BOOST_CHECK_EQUAL(yv.rows(), 3);
   BOOST_CHECK_EQUAL(yv.cols(), 2);
   BOOST_CHECK(yv == true_yv);
}

BOOST_AUTO_TEST_CASE(test_indexing)
{
  const auto N = 2;
  const auto C = 3;
  const auto H = 4;
  const auto W = 5;
  auto x = range(N * C * H * W).cast<float>().reshape(Vector4i{N, C, H, W});

  // This will fail if the reshape function is not defined for rvalue-reference
  // to MultiArray<T, N, O> objects.
  for (int i = 0; i < N * C * H * W; ++i)
    BOOST_CHECK_EQUAL(x.flat_array()(i), float(i));
};
