#include <gtest/gtest.h>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/FeatureDescriptors/Orientation.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestOrientation, test_orientation_histogram)
{
  const int N{ 5 };
  const int M{ N*N - 1 };

  auto grad_polar_coords = Image<Vector2f>{ N, N };
  const Point2f c{ grad_polar_coords.sizes().cast<float>() / 2.f };

  for (int gy = 0; gy < grad_polar_coords.height(); ++gy)
  {
    for (int gx = 0; gx < grad_polar_coords.width(); ++gx)
    {
      const auto theta = [&](){
        auto t = atan2(gy-c.y(), gx - c.x());
        if (t < 0)
          t += 2 * float(M_PI);
        return t;
      }();
      const auto theta_bin = int(floor(theta / float(2 * M_PI) * M)) % M;

      // Set all gradients to zero.
      for (int y = 0; y < grad_polar_coords.height(); ++y)
        for (int x = 0; x < grad_polar_coords.width(); ++x)
          grad_polar_coords(x,y) = Vector2f::Zero();

      // Except at coords (gx, gy).
      grad_polar_coords(gx, gy) =  Vector2f{ 1.f, theta };

      Array<float, M, 1> histogram;
      compute_orientation_histogram(histogram, grad_polar_coords,
                                    c.x(), c.y(), 1.f);
      histogram /= histogram.sum();

      Matrix<float, M, 1> expected_histogram;
      expected_histogram.setZero();
      expected_histogram[theta_bin] = 1.f;

      ASSERT_MATRIX_NEAR(expected_histogram, histogram.matrix(), 1e-6f);
    }
  }
}


TEST(TestComputeDominantOrientations, test_detect_single_peak)
{
  const int N{ 5 };

  auto grad_polar_coords = Image<Vector2f>{ N, N };
  const Point2f c{ grad_polar_coords.sizes().cast<float>() / 2.f };

  const Vector2f g{ Vector2f::Zero() };
  const auto theta = atan2(0 - c.y(), 0 - c.x());

  // Set all gradients to zero except at coords (gx, gy).
  for (int y = 0; y < grad_polar_coords.height(); ++y)
    for (int x = 0; x < grad_polar_coords.width(); ++x)
      grad_polar_coords(x,y) = Vector2f::Zero();
  grad_polar_coords(0, 0) =  Vector2f{ 1.f, theta };

  auto dominant_orientations = ComputeDominantOrientations{}(
    grad_polar_coords, c.x(), c.y(), 1.f
  );

  EXPECT_EQ(dominant_orientations.size(), 1u);

  auto& ori = dominant_orientations.front();
  EXPECT_NEAR(theta, ori, 1e-6f);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
