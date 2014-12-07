#include <DO/Geometry/Objects/Ellipse.hpp>
#include <DO/Geometry/Objects/Triangle.hpp>
#include <DO/Geometry/Tools/Utilities.hpp>

#include "TestPolygon.hpp"


using namespace DO;
using namespace std;


class TestEllipse : public TestPolygon
{
protected:
  TestEllipse() : TestPolygon()
  {
    _width = 300;
    _height = 300;
  }
};


TEST_F(TestEllipse, test_segment_area)
{
  Ellipse E(270, 150, to_radian(42.), Point2d::Zero());

  try
  {
    int steps = 18;

    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      for (int i1 = i0+1; i1 < i0+steps; ++i1)
      {
        double theta1 = i1*2*M_PI/steps;

        const Point2d a(E(theta0));
        const Point2d b(E(theta1));
        const Point2d& o = E.center();
        Triangle t(o, a, b);

        auto inside_segment = [&](const Point2d& p) -> bool {
          return (ccw(a,b,p) == -1) && inside(p, E);
        };

        double segArea1 = sweep_count_pixels(inside_segment);
        double segArea2 = segment_area(E, theta0, theta1);

        double absError = fabs(segArea1 -segArea2);
        double relError = absError/segArea1;

        const double thres = 0.1;
        EXPECT_NEAR(relError, 0, thres);
#if 0
        if (relError > thres)
        {
          printStage("Numerical error");
          CHECK(i0);
          CHECK(i1);
          CHECK(toDegree(theta0));
          CHECK(toDegree(theta1));
          CHECK(segArea1);
          CHECK(segArea2);
          CHECK(relError);
          getKey();
        }
#endif
      }
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }

}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}