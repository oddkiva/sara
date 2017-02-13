#include <gtest/gtest.h>

#include <DO/Sara/FeatureDetectors/Hessian.hpp>


using namespace DO::Sara;


TEST(TestHessianLaplaceDetector, test_detection)
{
  const auto N = 2 * 10 + 1;
  auto I = Image<float>{N, N};
  I.flat_array().fill(0);
  I(1, 1) = 1.f;

  ComputeHessianLaplaceMaxima compute_hessian_laplace_maxima{};

  auto features = compute_hessian_laplace_maxima(I, 0);
}

TEST(TestDoHDetector, test_detection)
{
  constexpr auto N = 2 * 10 + 1;
  auto I = Image<float>{N, N};
  I.flat_array().fill(0);
  I(1, 1) = 1.f;

  ComputeDoHExtrema compute_doh_maxima{};

  auto features = compute_doh_maxima(I, 0);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
