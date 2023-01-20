#define BOOST_TEST_MODULE "MultiViewGeometry/RANSAC Algorithm"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/FundamentalMatrixSolvers.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestRansac)

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
  BOOST_CHECK(samples.matrix().minCoeff() >= 0);
  BOOST_CHECK(samples.matrix().maxCoeff() < 10);
}

BOOST_AUTO_TEST_CASE(test_ransac_with_eight_point_algorithm)
{
  // Check this so that it can be serialized with HDF5.
  static_assert(sizeof(FundamentalMatrix{}) == sizeof(Eigen::Matrix3d{}));

  auto left = Tensor_<double, 2>{8, 3};
  auto right = Tensor_<double, 2>{8, 3};

  // clang-format off
  left.colmajor_view().matrix() <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826, 0.253816,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057, 0.675353,
           1,        1,        1,        1,        1,        1,        1,        1;

  right.colmajor_view().matrix() <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151, 0.530029,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756, 0.604387,
           1,        1,        1,        1,        1,        1,        1,        1;
  // clang-format on

  auto matches = Tensor_<int, 2>{8, 2};
  for (auto i = 0; i < 8; ++i)
    matches[i].flat_array() << i, i;

  const auto X = PointCorrespondenceList{matches, left, right};

  const auto iterations = 10;

  auto inlier_predicate = InlierPredicate<SampsonEpipolarDistance>{};
  inlier_predicate.err_threshold = 1e-3;

  ransac(X, EightPointAlgorithm{}, inlier_predicate, iterations);
}

BOOST_AUTO_TEST_CASE(test_ransac_with_seven_point_algorithm)
{
  // Check this so that it can be serialized with HDF5.
  static_assert(sizeof(FundamentalMatrix{}) == sizeof(Eigen::Matrix3d{}));

  auto left = Tensor_<double, 2>{8, 3};
  auto right = Tensor_<double, 2>{8, 3};

  // clang-format off
  left.colmajor_view().matrix() <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826, 0.253816,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057, 0.675353,
           1,        1,        1,        1,        1,        1,        1,        1;

  right.colmajor_view().matrix() <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151, 0.530029,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756, 0.604387,
           1,        1,        1,        1,        1,        1,        1,        1;
  // clang-format on

  auto matches = Tensor_<int, 2>{8, 2};
  for (auto i = 0; i < 8; ++i)
    matches[i].flat_array() << i, i;

  const auto X = PointCorrespondenceList{matches, left, right};

  const auto iterations = 10;

  auto inlier_predicate = InlierPredicate<SampsonEpipolarDistance>{};
  inlier_predicate.err_threshold = 1e-3;

  const auto [F, inliers, subset_best] = ransac(
      X, SevenPointAlgorithmDoublePrecision{}, inlier_predicate, iterations, std::nullopt, true);
}

BOOST_AUTO_TEST_SUITE_END()
