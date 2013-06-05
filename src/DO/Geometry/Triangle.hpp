/*
 * =============================================================================
 *
 *       Filename:  Triangle.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  29/06/2010 18:07:00
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_GEOMETRY_TRIANGLE_HPP
#define DO_GEOMETRY_TRIANGLE_HPP

namespace DO {

  // Triangle with vertices (a,b,c) in CCW order.
  class Triangle
  {
  public:
    Triangle() {}

    Triangle(const Point2d& a, const Point2d& b, const Point2d& c);

    double area() const;

    bool isInside(const Point2d& p) const;

    void drawOnScreen(const Rgb8& col = Red8) const;

  private:
    Point2d v[3]; // vertices
    Vector2d n[3]; // outward normals
  };


	//! Simple criterion to test if the triangle is too flat
	class TriangleFlatness
	{
	public:
		TriangleFlatness(double lowestAngleDegree, double secondLowestDegree)
		  : lb(std::cos(toRadian(lowestAngleDegree)))
      , lb2(std::cos(toRadian(secondLowestDegree))) {}

		inline bool operator()(const Point2d& a, const Point2d& b, const Point2d& c) const
		{ return !isNotFlat(a, b, c); }

    bool isNotFlat(const Point2d& a, const Point2d& b, const Point2d& c) const;

	private:
		const double lb;
		const double lb2;
	};

} /* namespace DO */

#endif /* DO_TRIANGLE_HPP */