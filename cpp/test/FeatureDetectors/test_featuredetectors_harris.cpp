#include <gtest/gtest.h>

#include <DO/Sara/FeatureDetectors/Harris.hpp>


using namespace DO::Sara;


TEST(TestHarris, test_scale_adapted_harris_corners)
{
  // TODO.
}

TEST(TestHarris, test_harris_cornerness_pyramid)
{
  // TODO.
}

TEST(TestHarris, test_me)
{
  const auto N = 2 * 10 + 1;
  Image<float> I{ N, N };
  I.array().fill(0);
  I(1, 1) = 1.f;

  ComputeHarrisLaplaceCorners compute_harris_laplace_corners{};

  auto features = compute_harris_laplace_corners(I, 0);

}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}