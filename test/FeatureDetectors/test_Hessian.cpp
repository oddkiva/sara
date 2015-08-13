#include <gtest/gtest.h>

#include <DO/Sara/FeatureDetectors/Hessian.hpp>


using namespace DO::Sara;


TEST(TestHessian, test_det_of_hessian_pyramid)
{
  EXPECT_FALSE(true);
}

TEST(TestHessian, test_me)
{
  const auto N = 2 * 10 + 1;
  Image<float> I{ N, N };
  I.array().fill(0);
  I(1, 1) = 1.f;

  ComputeHessianLaplaceMaxima compute_hessian_laplace_maxima{};

  auto features = compute_hessian_laplace_maxima(I, 0);

  EXPECT_FALSE(true);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}