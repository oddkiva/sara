#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>

namespace DO {

  Triangle::Triangle(const Point2d& a, const Point2d& b, const Point2d& c)
  {
    v[0] = a; v[1] = b; v[2] = c;
    Matrix2d U;
    U.col(0) = b-a;
    U.col(1) = c-a;
    if (U.determinant() < 0)
      std::swap(v[1], v[2]);

    for (int i = 0; i < 3; ++i)
    {
      n[i] = v[(i+1)%3] - v[i];
      std::swap(n[i].x(), n[i].y());
      n[i].y() = -n[i].y();
    }
  }

  double Triangle::area() const
  {
    Matrix2d M;
    M.col(0) = v[1]-v[0];
    M.col(1) = v[2]-v[0];
    return 0.5*abs(M.determinant());
  }

  bool Triangle::isInside(const Point2d& p) const
  {
    for (int i = 0; i < 3; ++i)
    {
      Vector2d u(p-v[i]);
      if (n[i].dot(u) > 1e-10)
        return false;
    }
    return true;
  }

  void Triangle::drawOnScreen(const Rgb8& col) const
  {
    drawLine(v[0],v[1],col);
    drawLine(v[1],v[2],col);
    drawLine(v[0],v[2],col);
  }
  
  bool TriangleFlatness::isNotFlat(const Point2d& a, const Point2d& b, 
                                   const Point2d& c) const
  {
    Vector2d d[3] = { (b-a), (c-a), (c-b) };
    for (int i = 0; i < 3; ++i)
      d[i].normalize();
    // Find the two smallest angles. They necessarily have non negative
    // dot products.
    double dot[3] = { d[0].dot(d[1]), d[1].dot(d[2]), -d[2].dot(d[0]) };
    // Sort dot products in increasing order.
    std::sort(dot, dot+3);
    // We need to consider two cases:
    // 1. All the dot products are non negative.
    //    Then the three angles are less than 90 degrees.
    // 2. One dot product is negative. It corresponds to the greatest 
    //    angle of the triangle.
    // In the end, the smallest angles have the greatest cosines which 
    // are in both cases dot[1] and dot[2] with dot[1] < dot[2].
    // In our case dot[1] <= cos(40°)=lb2 and dot[2] <= cos(30°)=lb1
    return (lb2 >= dot[1] && lb >= dot[2]);
  }
}