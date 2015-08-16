#include <gtest/gtest.h>

#include <DO/Sara/FeatureDetectors/LoG.hpp>


using namespace DO::Sara;


TEST(TestLoGExtrema, test_compute_LoG_extrema)
{
  const auto N = 2 * 5 + 1;
  Image<float> I{ N, N };
  I.array().fill(0);

  const auto xc = N / 2.f;
  const auto yc = N / 2.f;
  const auto sigma = 1.5f;
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x)
      I(x, y) = 1 / sqrt(2*float(M_PI)*pow(sigma, 2)) *
                exp(-(pow(x - xc, 2) + pow(y - yc, 2)) / (2 * pow(sigma, 2)));

  using namespace std;

  const auto pyramid_params = ImagePyramidParams{};
  auto compute_LoGs = ComputeLoGExtrema{ pyramid_params };

  auto scale_octave_pairs = vector<Point2i>{};
  auto features = compute_LoGs(I, &scale_octave_pairs);
  const auto& o_index = scale_octave_pairs[0](1);

  EXPECT_EQ(features.size(), 1);
  EXPECT_EQ(scale_octave_pairs.size(), 1);

  const auto& f = features.front();
  const auto& L = compute_LoGs.laplacians_of_gaussians();
  const auto z = L.octave_scaling_factor(o_index);

  EXPECT_NEAR(f.x()*z, xc, 1e-2);
  EXPECT_NEAR(f.y()*z, yc, 1e-2);
  EXPECT_NEAR(z, 0.5, 1e-2);
}

int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}