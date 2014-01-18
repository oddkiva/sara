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

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

// Simple polygon, i.e., polygon without holes
const std::size_t MaxVertices = 128;
typedef short_alloc<Point2d, MaxVertices> SmallPolyAllocator;
typedef std::vector<Point2d> SimplePolygon;

void drawPoly(const SimplePolygon& p, const Color3ub& color, int width = 3)
{
  for (size_t i = 0; i != p.size(); ++i)
    drawLine(p[i], p[(i+1)%p.size()], color, width);
}

// Utility function.
template <typename T>
inline int signum(T val)
{ return (T(0) < val) - (val < T(0)); }

// Check if matrix [u, v] forms a counter-clockwise oriented basis.
double cross(const Vector2d& u, const Vector2d& v)
{
  Matrix2d M;
  M.col(0) = u;
  M.col(1) = v;
  return M.determinant();
}

/*!
  Suppose line segment [a, b] is vertical.
  There are three cases:
  - If point 'c' in on the left, then det([b-a, c-a]) > 0.
  - If point 'c' in on the right, then det([b-a, c-a]) < 0.
  - If point 'c' in on the line (a,b), then det([b-a, c-a]) = 0.
  */
int ccw(const Point2d& a, const Point2d& b, const Point2d& c)
{ return signum(cross(b-a, c-a)); }

/*!
  Intersection test between lines [x0, x1] and [y0, y1].
  'u' is the intersection point if it exists.
 */
bool intersection(const Vector2d& s1, const Vector2d& e1,
                  const Vector2d& s2, const Vector2d& e2,
                  Vector2d& u)
{
  /*
    The intersection point 'u' is such that
    u = x0 + s(x1-x0)  (1)
    u = y0 + t(y1-y0)  (2)

    The goal is to determine the parameter 's' or 't'.
    It is sufficient to compute 's' for example.

    Using (1) = (2), it follows that
    s(x1-x0) - t(y1-y0) = y0-x0

    Using cross-product with (y1-y0), we have
    s (x1-x0)x(y1-y0) = (y0-x0)x(y1-y0)
    Thus
    s = (y0-x0)x(y1-y0) / (x1-x0)x(y1-y0)
   */

	Vector2d dx, dy, d;
	dx = e1 - s1;
	dy = e2 - s2;
	d  = s2 - s1;

  // Sanity check: lines must not be collinear.
	double dxy = cross(dx, dy);
  if (dxy == 0)
    return false;
	
  // Compute the parameter 's'.
  double s = cross(d, dy) / dxy;
 // // Check that 's' is in [0, 1], otherwise it is inside the segment [x0, x1].
	//if (s <= 0 || s >= 1)
 //   return false;
  
  // Plug parameter 's' back to the equation (1).
	u = s1 + s*dx;
	return true;
}

// Polygon 'subject' can be a polygon of any type.
// Polygon 'clip'  must be a convex polygon.
SimplePolygon sutherlandHodgman(const SimplePolygon& subject,
                                const SimplePolygon& clip)
{
  arena<MaxVertices> inArena, outArena;
  SimplePolygon in, out;
  Point2d inter;
  out = subject;

  const size_t M = clip.size();
  for (size_t e = 0; e != M; ++e) // 'e' like edge of the clip polygon.
  {
    in = out;
    out.clear();

    const size_t N = in.size();
    for (size_t v = 0; v != N; ++v)
    {
      int ccw_a = ccw(clip[e], clip[(e+1)%M], in[v]);
      int ccw_b = ccw(clip[e], clip[(e+1)%M], in[(v+1)%N]);

      if (ccw_a ==  1 && ccw_b ==  1)
        out.push_back(in[(v+1)%N]);
      else if (ccw_a ==  1 && ccw_b == -1)
      {
        if ( intersection(clip[e], clip[(e+1)%M],
                          in[v], in[(v+1)%N], inter) )
          out.push_back(inter);
      }
      else if (ccw_a == -1 && ccw_b ==  1)
      {
        if ( intersection(clip[e], clip[(e+1)%M],
                          in[v], in[(v+1)%N], inter) )
          out.push_back(inter);
        out.push_back(in[(v+1)%N]);
      }
    }
  }

  return out;
}

int main()
{
  int w, h;
  w = h = 400;

  HighResTimer timer;
  double elapsed;

  openWindow(w,h);
  setAntialiasing();

  SimplePolygon poly, clip, res;
  {
    int step = 18;
    for (int i = 0; i < step; ++i)
    {
      Point2d p;
      p << 
        w/2. + 100*cos(i*2*M_PI/step),
        h/2. + 150*sin(i*2*M_PI/step);
      poly.push_back(p);

      p.array() += 10.;
      clip.push_back(p);
    }
  }
  drawPoly(poly, Red8);
  drawPoly(clip, Blue8);
  getKey();

  int numIter = 1000;
  timer.restart();
  for (int i = 0; i < numIter; ++i)
    res = sutherlandHodgman(poly, clip);
  elapsed = timer.elapsed()/numIter;
  cout << "Intersection computation time = " << elapsed << " seconds" << endl;
 
  drawPoly(res, Green8,5);
  getKey();

  return 0;
}