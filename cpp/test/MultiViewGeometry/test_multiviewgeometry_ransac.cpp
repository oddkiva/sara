#define BOOST_TEST_MODULE "MultiViewGeometry/RANSAC Algorithm"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/FundamentalMatrixEstimators.hpp>
#include <DO/Sara/MultiViewGeometry/RANSAC.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


using namespace DO::Sara;
using namespace std;


BOOST_AUTO_TEST_SUITE(TestRansac)

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



BOOST_AUTO_TEST_CASE(test_ransac_with_eight_point_algorithm)
{
  // Check this so that it can be serialized with HDF5.
  static_assert(sizeof(FundamentalMatrix{}) == sizeof(Eigen::Matrix3d{}));

  auto left = Tensor_<double, 2>{8, 3};
  auto right = Tensor_<double, 2>{8, 3};

  left.colmajor_view().matrix() <<
    0.494292, 0.449212, 0.513487, 0.474079, 0.468652, 0.442959, 0.276826, 0.253816,
    0.734069, 0.595362, 0.685816,  0.58693, 0.689338, 0.577366, 0.117057, 0.675353,
           1,        1,        1,        1,        1,        1,        1,        1;

  right.colmajor_view().matrix() <<
    0.792952, 0.734874, 0.814332, 0.763281,   0.7605, 0.727001, 0.537151, 0.530029,
    0.644436, 0.515263, 0.596448, 0.504156, 0.603078, 0.498954, 0.115756, 0.604387,
           1,        1,        1,        1,        1,        1,        1,        1;

  auto matches = Tensor_<int, 2>{8, 2};
  for (auto i = 0; i < 8; ++i)
    matches[i].flat_array() << i, i;

  auto f_estimator = EightPointAlgorithm{};
  auto distance = [](const FundamentalMatrix& F, const Vector3d& left,
                     const Vector3d& right) {
    return double(right.transpose() * F.matrix() * left);
  };


  ransac(matches, left, right, f_estimator, distance, 10, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()
