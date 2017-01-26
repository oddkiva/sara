#include <DO/Sara/Core.hpp>
#include <DO/Sara/Geometry/Objects/Cone.hpp>
#include <DO/Sara/Geometry/Objects/CSG.hpp>
#include <DO/Sara/Geometry/Objects/Ellipse.hpp>
#include <DO/Sara/Geometry/Objects/Triangle.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>

#include "../AssertHelpers.hpp"

#include "TestPolygon.hpp"


using namespace std;
using namespace DO::Sara;


class TestEllipse : public TestPolygon
{
protected:
  TestEllipse() : TestPolygon()
  {
    _width = 300;
    _height = 300;
  }

protected:
  double _a{ 250. };
  double _b{ 150. };
  double _theta_degree{ 30. };
};


TEST_F(TestEllipse, test_rho)
{
  const auto E = Ellipse{ _a, _b, to_radian(_theta_degree), _center };

  // Change of variable in polar coordinates.
  // x = rho * cos(theta)
  // y = rho * sin(theta)
  auto expected_rho = [&](double theta) {
    const auto c = cos(theta);
    const auto s = sin(theta);
    const auto r = (_a*_b) / sqrt(_b*_b*c*c + _a*_a*s*s);
    return Vector2d{ r * unit_vector2(theta) };
  };

  const auto N = 10;
  for (int i = 0; i < N; ++i)
  {
    const auto theta = 2 * M_PI * i / N;
    ASSERT_MATRIX_NEAR(E.rho(theta), expected_rho(theta), 1e-8);
  }
}

TEST_F(TestEllipse, test_overloaded_operator)
{
  const auto E = Ellipse{ _a, _b, to_radian(_theta_degree), _center };

  // Change of variable in polar coordinates.
  // x = rho * cos(theta)
  // y = rho * sin(theta)
  auto expected_rho = [&](double theta) {
    const auto c = cos(theta);
    const auto s = sin(theta);
    const auto rho = (_a*_b) / sqrt(_b*_b*c*c + _a*_a*s*s);
    const Matrix2d R{ rotation2(to_radian(_theta_degree)) };
    return Vector2d{
      _center + R * rho * unit_vector2(theta)
    };
  };

  const auto N = 10;
  for (int i = 0; i < N; ++i)
  {
    const auto theta = 2 * M_PI * i / N;
    ASSERT_MATRIX_NEAR(E(theta), expected_rho(theta), 1e-8);
  }
}

TEST_F(TestEllipse, test_orientation)
{
  const auto E = Ellipse{ _a, _b, to_radian(_theta_degree), _center };

  const auto N = 10;
  for (int i = 0; i < N; ++i)
  {
    const auto theta = -M_PI + 2 * M_PI * i / N;
    const Point2d p{ _center + unit_vector2(theta + to_radian(_theta_degree)) };
    ASSERT_NEAR(orientation(p, E), theta,  1e-8);
  }
}

TEST_F(TestEllipse, test_ellipse_ostream)
{
  const auto E = Ellipse{ 250, 150, to_radian(_theta_degree), _center };

  stringstream buffer;
  CoutRedirect cout_redirect{ buffer.rdbuf() };
  cout << E << endl;

  auto text = buffer.str();

  EXPECT_NE(text.find("a = "), string::npos);
  EXPECT_NE(text.find("b = "), string::npos);
  EXPECT_NE(text.find("o = "), string::npos);
  EXPECT_NE(text.find("c = "), string::npos);
}

TEST_F(TestEllipse, test_sector_area)
{
  Ellipse E{ _a, _b, to_radian(_theta_degree), _center };

  CSG::Singleton<Ellipse> ell{ E };

  try
  {
    const auto steps = 18;
    const auto i0 = 0;
    const auto theta_0 = 2*M_PI*i0 / steps;
    const auto num_tests = 2;

    for (int i1 = i0 + 1; i1 < i0 + num_tests; ++i1)
    {
      auto theta_1 = 2*M_PI*i1  / steps;

      // Compute the sector area in a closed form.
      double analytic_sector_area = sector_area(E, theta_0, theta_1);

      // Build constructive solid geometry.
      CSG::Singleton<AffineCone2> cone{ affine_cone2(
        theta_0 + E.orientation(), theta_1 + E.orientation(), E.center())
      };
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
      auto estimated_sector_area = double{};
      if (i1 - i0 < steps/2)
        estimated_sector_area = sweep_count_pixels(inside_E_and_Cone);
      else if (abs(i1 - i0) == steps / 2)
        estimated_sector_area = area(E) / 2;
      else
        estimated_sector_area = sweep_count_pixels(inside_E_minus_Cone);

      // Absolute error and relative error.
      const auto abs_error = fabs(estimated_sector_area - analytic_sector_area);
      const auto rel_error = abs_error / estimated_sector_area;
      const auto thres = 1e-1;
      ASSERT_NEAR(rel_error, 0, thres);
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }
}

TEST_F(TestEllipse, test_segment_area)
{
  const auto E = Ellipse{ _a, _b, to_radian(_theta_degree), _center };

  try
  {
    const auto steps = 18;
    const auto i0 = 0;
    const auto theta_0 = 2*M_PI*i0 / steps;
    const auto num_tests = 2;

    for (int i1 = i0 + 1; i1 < i0 + num_tests; ++i1)
    {
      const auto theta_1 = 2*M_PI*i1 / steps;

      const Point2d a{ E(theta_0) };
      const Point2d b{ E(theta_1) };

      auto inside_segment = [&](const Point2d& p) {
        return (ccw(a,b,p) == -1) && E.contains(p);
      };

      const auto seg_area1 = sweep_count_pixels(inside_segment);
      const auto seg_area2 = segment_area(E, theta_0, theta_1);

      const auto abs_error = fabs(seg_area1 - seg_area2);
      const auto rel_error = abs_error / seg_area1;

      const auto thres = 0.1;
      ASSERT_NEAR(rel_error, 0, thres);
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }

}

TEST_F(TestEllipse, test_construct_from_shape_matrix)
{
  auto E = construct_from_shape_matrix(Matrix2d::Identity(), Vector2d::Zero());
  EXPECT_MATRIX_NEAR(shape_matrix(E), Matrix2d::Identity(), 1e-8);
  EXPECT_NEAR(E.radius1(), 1., 1e-8);
  EXPECT_NEAR(E.radius2(), 1., 1e-8);
  EXPECT_MATRIX_NEAR(E.center(), Point2d::Zero(), 1e-8);
}

TEST_F(TestEllipse, test_oriented_bbox)
{
  const auto E = Ellipse{ _a, _b, 0, Point2d::Zero() };
  const auto actual_bbox = oriented_bbox(E);
  const auto expected_bbox = Quad{ BBox{ -Point2d{ _a, _b }, Point2d{ _a, _b } } };
  for (int i = 0; i < 4; ++i)
    ASSERT_MATRIX_NEAR(expected_bbox[i], actual_bbox[i], 1e-8);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
