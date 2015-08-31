#include <gtest/gtest.h>

#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>

#include "../AssertHelpers.hpp"


using namespace std;
using namespace DO::Sara;


TEST(TestConvexHull, test_quad)
{
  auto points = vector<Point2d>{
    Point2d{ 0., 0. },
    Point2d{ 1., 0. },
    Point2d{ 1., 1. },
    Point2d{ 0., 1. },
    Point2d{ 0.5, 0.5 },
    Point2d{ 0.25, 0.25 }
  };

  auto expected_points = vector<Point2d>{
    Point2d{ 0., 0. },
    Point2d{ 1., 0. },
    Point2d{ 1., 1. },
    Point2d{ 0., 1. }
  };

  auto convex_hull = graham_scan_convex_hull(points);
  EXPECT_ITEMS_EQ(convex_hull, expected_points);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
