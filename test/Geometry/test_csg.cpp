#include <DO/Sara/Geometry/Objects/Cone.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Geometry/Objects/CSG.hpp>

#include "TestPolygon.hpp"


using namespace DO;
using namespace std;


class TestCSG : public TestPolygon
{
protected:
  TestCSG() : TestPolygon()
  {
    _width = 15;
    _height= 15;
  }
};


TEST_F(TestCSG, test_intersection_ellipse_cone)
{
  Ellipse E(12., 12., 0., Point2d::Zero());
  AffineCone2 K(Point2d(1,0), Point2d(0,1), Point2d::Zero());

  CSG::Singleton<Ellipse> ell(E);
  CSG::Singleton<AffineCone2> cone(K);
  auto inter = ell*cone;

  int estimatedArea = sweep_count_pixels(
    [&](const Point2d& p) {
      return inter.contains(p);
  });

  double trueArea = area(E)/4.;
  double relativeError = fabs(estimatedArea - trueArea)/trueArea;
  EXPECT_NEAR(relativeError, 0, 1.5e-1);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}