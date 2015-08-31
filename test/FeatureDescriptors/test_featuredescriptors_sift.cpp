#include <gtest/gtest.h>

#include <DO/Sara/FeatureDescriptors/SIFT.hpp>


using namespace std;
using namespace DO::Sara;


TEST(TestSIFTDescriptors, test_me)
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

  auto feature = OERegion{ c, 1.f };
  auto sift = ComputeSIFTDescriptor<>{}(feature, grad_polar_coords);

  // TODO.
}


int main(int argc, char *argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
