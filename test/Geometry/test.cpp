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

#include <gtest/gtest.h>
#include <DO/Core/DebugUtilities.hpp>
#include <DO/Core/Timer.hpp>
#include <DO/Graphics.hpp>
#include <DO/Geometry/BBox.hpp>
#include <DO/Geometry/Quad.hpp>
#include <DO/Geometry/Triangle.hpp>
#include <DO/Geometry/Graphics/DrawPolygon.hpp>
#include <DO/Geometry/EllipseIntersection.hpp>
#include <ctime>

using namespace std;
using namespace DO;

const int w = 512, h = 512;

void initRandom()
{ srand(static_cast<unsigned int>(time(0))); }

inline double myRandom(double a, double b)
{ return a + (b-a)*double(rand())/(RAND_MAX+1e-1) ; }

inline Point2d randPoint2d(double w, double h)
{ return Point2d(myRandom(0, w), myRandom(0, h)); }

Ellipse randomEllipse(double w, double h)
{
  return Ellipse(myRandom(0., w/2.), myRandom(0., w/2.),
                 myRandom(0,2*M_PI),
                 Point2d(myRandom(w/4., 3*w/4.), myRandom(h/4., 3*h/4.)) );
}

void testQuadAlgorithms()
{
  BBox b(Point2d(w/4, h/4), Point2d(3*w/4, 3*h/4));
  Quad quad(b);

  do
  {
    if (!getActiveWindow())
      openWindow(w,h);
    clearWindow();

    Point2d p(randPoint2d(w, h));
    drawQuad(quad, Blue8);
    fillCircle(p.cast<float>(), 3.0f, Red8);
    if ( inside(p, quad) )
      cout << "p is inside quad" << endl;
    else
      cout << "p is not inside quad" << endl;
    
    Quad quad2(BBox(Point2d::Zero(), p));
    drawQuad(quad2, Red8);
    //cout << "There is "
    //  << (intersection(quad, quad2) ? "" : "no ") << "intersection." << endl;

  } while(getKey() != KEY_ESCAPE);
}

#ifdef DEBUG
void testAffineTransforms()
{
  Point2f c(w/2., h/2.);
  double r = 100.;
  int N = 100;
  MatrixXf p(2, N);
  for (int i = 0; i < N; ++i)
  {
    float theta = 2.*M_PI*i/N;
    p.col(i) = Point2f(cos(theta), sin(theta));
  }

  Matrix2f R;
  R << 0.5, 2,
         0, 1;

  for (int i = 0; i < N; ++i)
    fillCircle(c+r*p.col(i), 3.f, Blue8);
  for (int i = 0; i < N; ++i)
    fillCircle(c+r*R*p.col(i), 3.f, Red8);
  getKey();

  Matrix2f S = R.transpose()*R;
  cout << S << endl;

  Ellipse E(fromShapeMat(S.cast<double>(), c.cast<double>()));
  E.drawOnScreen(Green8);
  getKey();
}
#endif

TEST(DO_Geometry_Test, bboxTest)
{
  BBox bbox(Point2d(w/4, h/4), Point2d(3*w/4, 3*h/4));
  EXPECT_EQ(bbox.topLeft(), Point2d(w/4, h/4));
  EXPECT_EQ(bbox.topRight(), Point2d(3*w/4, h/4));
  EXPECT_EQ(bbox.bottomRight(), Point2d(3*w/4, 3*h/4));
  EXPECT_EQ(bbox.bottomLeft(), Point2d(w/4, 3*h/4));

  //if (!getActiveWindow())
  //  setAntialiasing(openWindow(w,h));
  //drawBBox(b, Blue8, 3);

  Point2d points[] = {
    Point2d::Zero(),
    Point2d(w/4, w/4),
    Point2d(w/2., h/2.)
  };
  EXPECT_FALSE(inside(points[0], bbox));
  EXPECT_TRUE (inside(points[1], bbox));
  EXPECT_TRUE (inside(points[2], bbox));

  bbox = BBox(points, points+3);
  EXPECT_EQ(bbox.topLeft(), points[0]);
  EXPECT_EQ(bbox.bottomRight(), points[2]);

  // \todo: intersection test.
}

void compareComputationTimes(const Ellipse& e1, const Ellipse& e2,
                             int discretization)
{
  HighResTimer t;
  int times = 1e3;
  // Approximate computation.
  t.restart();
  double approxRatio, analyticRatio;
  for (int i = 0; i < times; ++i)
    approxRatio = approximateIntersectionUnionRatio(e1, e2, discretization);
  double approximateTime = t.elapsedMs()/double(times);
  cout << "Approximate computation time = " << approximateTime << " ms" << endl;
  
  t.restart();
  for (int i = 0; i < times; ++i)
    analyticRatio = analyticInterUnionRatio(e1, e2);
  double closedFormTime = t.elapsedMs()/double(times);
  cout << "Closed-form computation time = " << closedFormTime << " ms" << endl;
  
  //
  double speedGain = approximateTime / closedFormTime;
  cout << "speed gain = " << speedGain << endl;
}

void viewEllipses(const Ellipse& e1, const Ellipse& e2)
{
  // Verbose comments.
  cout << endl;
  cout << "Ellipse 1" << endl;
  cout << e1 << endl;
  cout << "Ellipse 2" << endl;
  cout << e2 << endl;
  
  if (!getActiveWindow())
    setAntialiasing(openWindow(w, h));
  
  clearWindow();
  drawEllipse(e1, Red8, 3);
  drawEllipse(e2, Blue8, 3);
  
  Point2d interPts[4];
  int numInter = computeEllipseIntersections(interPts, e1, e2);
  cout << "\nIntersection count = " << numInter << endl;
  for (int i = 0; i < numInter; ++i)
  {
    fillCircle(interPts[i].cast<float>(), 5.f, Green8);
    cout << "[" << i << "] " << interPts[i].transpose() << endl;
  }
}

TEST(DO_Geometry_Test, ellipseAlgorithmsTest)
{
  int discretization = 36;
  do
  {
    Ellipse e1(randomEllipse(w, h));
    Ellipse e2(randomEllipse(w, h));

    // Visualize.
    bool display = false;
    if (display)
      viewEllipses(e1, e2);

    // ====================================================================== //
    // Approximate computation of intersecting ellipses.
    //
    // Very reliable and based on Sutherland-Hodgman clipping method.
    // Approximating an ellipse by an 8-sided polygon turns out to be actually 
    // a quite good approximation in practice!
    // We use in our method.
    //
    // For a good trade-off between precision and speed, it seems sufficient to
    // choose a 36-sided polygon in practice.
    double approxRatio = approximateIntersectionUnionRatio(e1, e2, discretization);

    // ====================================================================== //
    // Closed-form computation.
    // The computation is really good but...
    // \todo: there are still numerical precision issues to investigate.
    // Numerical precision are tuned by hand for now...
    double analyticRatio = analyticInterUnionRatio(e1, e2);

    // ====================================================================== //
    // Speed comparison.
    // 
    // Experimental settings.
    bool compareSpeed = false;
    if (compareSpeed)
      compareComputationTimes(e1, e2, discretization);

    // Summary
    double relativeError = abs(approxRatio-analyticRatio) / approxRatio;
    double absoluteError = abs(approxRatio-analyticRatio);
    
    double areaRatio = min(area(e1), area(e2)) / max(area(e1), area(e2));
    if (absoluteError > 0.05  && areaRatio > 0.2)
    {
      viewEllipses(e1, e2);
      CHECK(area(e1));
      CHECK(area(e2));
      CHECK(inside(e2.c(), e1));
      CHECK(inside(e1.c(), e2));
      CHECK(analyticInterArea(e1, e2, true));
      cout << "Approximate intersection-union area ratio = " << approxRatio << endl;
      cout << "Exact intersection-union area ratio = " << analyticRatio << endl;
      cout << "Relative error = " << relativeError << endl;
      cout << "Absolute error = " << absoluteError << endl;
      getKey();
    }
  } while(true);
}

TEST(DO_Geometry_Test, quadTest)
{
  BBox bbox(Point2d(w/4, h/4), Point2d(3*w/4, 3*h/4));
  EXPECT_EQ(bbox.topLeft(), Point2d(w/4, h/4));
  EXPECT_EQ(bbox.topRight(), Point2d(3*w/4, h/4));
  EXPECT_EQ(bbox.bottomRight(), Point2d(3*w/4, 3*h/4));
  EXPECT_EQ(bbox.bottomLeft(), Point2d(w/4, 3*h/4));

  Quad q(bbox);
  EXPECT_NEAR(area(bbox), area(q), 1e-10);

  Triangle t1(Point2d(0,0), Point2d(100, 0), Point2d(100, 100));
  EXPECT_NEAR(area(t1), 1e4/2., 1e-10);

  Triangle t2(Point2d(100,0), Point2d(0, 0), Point2d(100, 100));
  EXPECT_NEAR(signedArea(t2), -1e4/2., 1e-10);
  
  Point2d  p1(w/2., h/2.);
  EXPECT_TRUE(inside(p1, q));
  
  testQuadAlgorithms();
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}