// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

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