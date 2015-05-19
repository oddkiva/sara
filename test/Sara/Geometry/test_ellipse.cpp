#include <DO/Sara/Core.hpp>
#include <DO/Sara/Geometry/Objects/Cone.hpp>
#include <DO/Sara/Geometry/Objects/CSG.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Geometry/Objects/Triangle.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>

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


// TODO: make test faster. TOO SLOW!!!
TEST_F(TestEllipse, DISABLED_test_sector_area)
{
  Ellipse E(250, 150, to_radian(75.), _center);

  CSG::Singleton<Ellipse> ell(E);

  try {

    int steps = 18;
    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      for (int i1 = i0+1; i1 <= i0+steps; ++i1)
      {
        double theta1 = i1*2*M_PI/steps;

        // Compute the sector area in a closed form.
        double analytic_sector_area = sector_area(E, theta0, theta1);

        // Build constructive solid geometry.
        CSG::Singleton<AffineCone2> cone ( affine_cone2(
          theta0+E.orientation(), theta1+E.orientation(), E.center()) );
        auto E_and_Cone = ell*cone;
        auto E_minus_Cone = ell - E_and_Cone;

        // Use the CSGs to estimate the sector area by pixel counting.
        auto inside_E_and_Cone = [&](const Point2d& p){
          return E_and_Cone.contains(p);
        };
        auto inside_E_minus_Cone = [&](const Point2d& p){
          return E_minus_Cone.contains(p);
        };

        // Use the lambda functors to estimate the elliptic sector area.
        double estimated_sector_area;
        if (i1 - i0 < steps/2)
          estimated_sector_area = sweep_count_pixels(inside_E_and_Cone);
        else if (abs(i1-i0) == steps/2)
          estimated_sector_area = area(E) / 2.;
        else
          estimated_sector_area = sweep_count_pixels(inside_E_minus_Cone);

        // Absolute error and relative error.
        double abs_error = fabs(estimated_sector_area - analytic_sector_area);
        double rel_error = abs_error / estimated_sector_area;

        double thres = 1e-1;
        EXPECT_NEAR(rel_error, 0, thres);
        if (rel_error > thres)
        {
          print_stage("Numerical error");
          cout << i0 << "     " << i1 << endl;
          CHECK(abs(i1-i0));
          CHECK(analytic_sector_area);
          CHECK(estimated_sector_area);
          CHECK(abs_error);
          CHECK(rel_error);
        }
      }
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }
}

// TODO: make test faster. TOO SLOW!!!
TEST_F(TestEllipse, DISABLED_test_segment_area)
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
        if (relError > thres)
        {
          print_stage("Numerical error");
          CHECK(i0);
          CHECK(i1);
          CHECK(to_degree(theta0));
          CHECK(to_degree(theta1));
          CHECK(segArea1);
          CHECK(segArea2);
          CHECK(relError);
        }
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