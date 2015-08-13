#include <gtest/gtest.h>

#include <DO/Sara/FeatureDetectors/DoG.hpp>


using namespace DO::Sara;


TEST(TestExtremumRefinement, test_on_edge)
{
  EXPECT_FALSE(true);
}

TEST(TestExtremumRefinement, test_refine_extremum)
{
  EXPECT_FALSE(true);
}

TEST(TestExtremumRefinement, test_local_scale_space_extrema)
{
  EXPECT_FALSE(true);
}

TEST(TestDoG, test_me)
{
  const auto N = 2 * 10 + 1;
  Image<float> I{ N, N };
  I.array().fill(0);
  I(1, 1) = 1.f;

  ComputeDoGExtrema compute_DoGs{};

  auto features = compute_DoGs(I, 0);

  EXPECT_FALSE(true);
}


int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}