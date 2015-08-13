#include <gtest/gtest.h>

#include <DO/Sara/FeatureDetectors/LoG.hpp>


using namespace DO::Sara;


TEST(TestLoGs, test_me)
{
  const auto N = 2 * 10 + 1;
  Image<float> I{ N, N };
  I.array().fill(0);
  I(1, 1) = 1.f;

  ComputeLoGExtrema compute_LoGs{};

  auto features = compute_LoGs(I, 0);

  EXPECT_FALSE(true);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}