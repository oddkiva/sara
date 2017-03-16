#define BOOST_TEST_MODULE "FeatureDescriptors/Harris Affine Detector"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/FeatureDetectors/Harris.hpp>


using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestHarrisAffineDetector)

BOOST_AUTO_TEST_CASE(test_scale_adapted_harris_corners)
{
  // TODO.
}

BOOST_AUTO_TEST_CASE(test_harris_cornerness_pyramid)
{
  // TODO.
}

BOOST_AUTO_TEST_CASE(test_me)
{
  constexpr auto N = 2 * 10 + 1;
  auto I = Image<float>{N, N};
  I.flat_array().fill(0);
  I(1, 1) = 1.f;

  ComputeHarrisLaplaceCorners compute_harris_laplace_corners{};

  auto features = compute_harris_laplace_corners(I, 0);
}

BOOST_AUTO_TEST_SUITE_END()