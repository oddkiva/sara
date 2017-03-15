#include <gtest/gtest.h>

#include <DO/Sara/Geometry/Tools/Metric.hpp>

#include "../AssertHelpers.hpp"


using namespace DO::Sara;

typedef testing::Types<
  SquaredRefDistance<float, 2>,
  SquaredDistance<float, 2>
> DistanceTypes;

template <typename DistanceType>
class TestSquaredDistanceAndOpenBall : public testing::Test
{
};

TYPED_TEST_CASE_P(TestSquaredDistanceAndOpenBall);
TYPED_TEST_P(TestSquaredDistanceAndOpenBall, test_computations)
{
  using Distance = TypeParam;
  static_assert(Distance::Dim == 2, "Wrong dimension");

  Matrix2f A = Matrix2f::Identity();

  Vector2f a = Vector2f::Zero();
  Vector2f b = Vector2f{ 1.f, 0.f };
  Vector2f c = Vector2f{ 0.f, 1.f };

  Distance d{ A };
  EXPECT_MATRIX_EQ(d.covariance_matrix(), A);
  EXPECT_TRUE(d.is_quasi_isotropic());
  EXPECT_NEAR(d(b, a), A(0, 0), 1e-6f);
  EXPECT_NEAR(d(c, a), A(1, 1), 1e-6f);

  OpenBall<Distance> ball(Point2f::Zero(), 1.1f, d);
  EXPECT_MATRIX_EQ(ball.center(), Point2f::Zero());
  EXPECT_EQ(ball.radius(), 1.1f);
  EXPECT_EQ(ball.squared_distance().covariance_matrix(), A);
  EXPECT_TRUE(ball.contains(a));
  EXPECT_TRUE(ball.contains(b));
  EXPECT_TRUE(ball.contains(c));
}

REGISTER_TYPED_TEST_CASE_P(
  TestSquaredDistanceAndOpenBall,
  test_computations);
INSTANTIATE_TYPED_TEST_CASE_P(Geometry_Tools_Metric,
                              TestSquaredDistanceAndOpenBall,
                              DistanceTypes);

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
