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
#include <DO/Graphics.hpp>
#include <DO/Core/DebugUtilities.hpp>
#include <DO/Core/Timer.hpp>
#include <DO/Graphics.hpp>
#include <DO/Geometry/Objects.hpp>
#include <DO/Geometry/Graphics.hpp>
#include <DO/Geometry/Algorithms/EllipseIntersection.hpp>
#include <DO/Geometry/Objects/Cone.hpp>
#include <DO/Geometry/Tools/Utilities.hpp>
#include <ctime>

using namespace std;
using namespace DO;

namespace TestParams
{
  const int w = 512;
  const int h = w;
  const double a = 0.25*w, b = 0.75*h;
  const Point2d p1(a,a), p2(b,b), center(w/2., h/2.);
  const bool debug = true;

  void initRandom()
  { srand(static_cast<unsigned int>(time(0))); }

  inline double random(double a, double b)
  { return a + (b-a)*double(rand())/(RAND_MAX+1e-1) ; }

  inline Point2d randPoint2d(double w, double h)
  { return Point2d(random(0, w), random(0, h)); }

  Ellipse randomEllipse(double w, double h)
  {
    return Ellipse(random(0., w/2.), random(0., w/2.),
                   random(0,2*M_PI),
                   Point2d(random(w/4., 3*w/4.), random(h/4., 3*h/4.)) );
  }

} /* namespace TestParameters */

template <typename TestPred, typename GroundTruth>
void sweepTest(const TestPred& pred,
               const GroundTruth& gdTruth)
{
  if (TestParams::debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  for (int y = 0; y < TestParams::h; ++y)
  {
    for (int x = 0; x < TestParams::w; ++x)
    {
      Point2d p(x,y);
      EXPECT_EQ(gdTruth(p), pred(p));

      if (pred(p) && gdTruth(p) && TestParams::debug)
        drawPoint(x,y,Green8);
      if (pred(p) != gdTruth(p) && TestParams::debug)
      {
        cout << "Faulty point" << endl;
        CHECK(p.transpose());
        fillCircle(x,y,10,Red8);
        getKey();
      }
    }
  }
}

template <typename TestPred>
int sweepCountPixels(const TestPred& pred)
{
  if (TestParams::debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  int quantity = 0;
  for (int y = 0; y < TestParams::h; ++y)
  {
    for (int x = 0; x < TestParams::w; ++x)
    {
      Point2d p(x,y);
      if (pred(p))
      {
        ++quantity;
        if (TestParams::debug)
          drawPoint(x,y, Green8);
      }
    }
  }

  if (TestParams::debug)
    getKey();

  return quantity;
}

template <typename TestPred>
int sweepCountPixels(const TestPred& pred, Image<Rgb8>& buffer)
{
  buffer.array().fill(White8);
  int interArea = 0;
  for (int y = 0; y < TestParams::h; ++y)
  {
    for (int x = 0; x < TestParams::w; ++x)
    {
      Point2d p(x,y);
      if (pred(p))
      {
        ++interArea;
        buffer(x,y) = Red8;
      }
    }
  }
  return interArea;
}

#ifdef THESE_ARE_OK
TEST(DO_Geometry_Test, bboxTest)
{
  BBox bbox(TestParams::p1, TestParams::p2);
  EXPECT_EQ(bbox.topLeft(), Point2d(TestParams::a, TestParams::a));
  EXPECT_EQ(bbox.topRight(), Point2d(TestParams::b, TestParams::a));
  EXPECT_EQ(bbox.bottomRight(), Point2d(TestParams::b, TestParams::b));
  EXPECT_EQ(bbox.bottomLeft(), Point2d(TestParams::a, TestParams::b));

  auto predicate = [&](const Point2d& p) {
    return inside(p, bbox);
  };
  auto groundTruth = [&](const Point2d& p) {
    return 
      p.cwiseMin(TestParams::p1) == TestParams::p1 && 
      p.cwiseMax(TestParams::p2) == TestParams::p2;
  };
  sweepTest(predicate, groundTruth);


  Point2d points[] = {
    Point2d::Zero(),
    Point2d(TestParams::a, TestParams::a),
    TestParams::center
  };
  EXPECT_FALSE(inside(points[0], bbox));
  EXPECT_TRUE (inside(points[1], bbox));
  EXPECT_TRUE (inside(points[2], bbox));

  bbox = BBox(points, points+3);
  EXPECT_EQ(bbox.topLeft(), points[0]);
  EXPECT_EQ(bbox.bottomRight(), points[2]);
}

// \todo: debug this
TEST(DO_Geometry_Test, quadTest)
{
  BBox bbox(TestParams::p1, TestParams::p2);
  Quad quad(bbox);

  EXPECT_NEAR(area(bbox), area(quad), 1e-10);

  //auto predicate = [&](const Point2d& p) {
  //  return inside(p, quad);
  //};
  //auto groundTruth = [&](const Point2d& p) {
  //  return 
  //    (p.cwiseMin(TestParams::p1) - TestParams::p1).norm() < 1e-1 && 
  //    (p.cwiseMax(TestParams::p2) - TestParams::p2).norm() < 1e-1;
  //};
  //sweepTest(predicate, groundTruth);
}

TEST(DO_Geometry_Test, triangleTest)
{
  Triangle t1(Point2d(0,0), Point2d(100, 0), Point2d(100, 100));
  EXPECT_NEAR(area(t1), 1e4/2., 1e-10);

  Triangle t2(Point2d(100,0), Point2d(0, 0), Point2d(100, 100));
  EXPECT_NEAR(signedArea(t2), -1e4/2., 1e-10);

  Triangle t3(Point2d(50, 73), Point2d(350, 400), Point2d(25, 200));
  int pixelArea3 = sweepCountPixels([&](Point2d& p){ return inside(p, t3); });
  double exactArea3 = area(t3);

  double relError = fabs(exactArea3 - pixelArea3)/exactArea3;
  EXPECT_NEAR(relError, 0., 5e-2);
}

TEST(DO_Geometry_Test, coneTest)
{
  Vector2d alpha(1,0), beta(1,1); // generators of the cone
  AffineCone2 convexK(alpha, beta, TestParams::center, AffineCone2::Convex, std::numeric_limits<double>::epsilon());
  AffineCone2 bluntK(alpha, beta, TestParams::center, AffineCone2::Blunt, std::numeric_limits<double>::epsilon());
  AffineCone2 convexPointedK(alpha, alpha, TestParams::center, AffineCone2::Convex);
  AffineCone2 bluntPointedK(alpha, alpha, TestParams::center, AffineCone2::Blunt, std::numeric_limits<double>::epsilon());
  AffineCone2 convexPointedK2(alpha, -alpha, TestParams::center, AffineCone2::Convex, std::numeric_limits<double>::epsilon());
  AffineCone2 bluntPointedK2(alpha, -alpha, TestParams::center, AffineCone2::Blunt, std::numeric_limits<double>::epsilon());

  // ======================================================================== //
  printStage("Convex affine cone testing");
  auto convexPredicate = [&](const Point2d& p) {
    return inside(p, convexK);
  };
  auto convexGroundTruth = [&](const Point2d& p) {
    return
      p.x() > TestParams::w/2. &&
      p.y() > TestParams::h/2. && 
      p.x() > p.y();
  };
  sweepTest(convexPredicate, convexGroundTruth);
  if (TestParams::debug)
  {
    drawAffineConeAxes(convexK);
    getKey();
  }

  // ======================================================================== //
  printStage("Blunt affine cone testing");
  auto bluntPredicate = [&](const Point2d& p) {
    return inside(p, bluntK);
  };
  auto bluntGroundTruth = [&](const Point2d& p) {
    return
      p.x() >= TestParams::w/2. && 
      p.y() >= TestParams::h/2. && 
      p.x() >= p.y();
  };
  sweepTest(bluntPredicate, bluntGroundTruth);
  if (TestParams::debug)
  {
    drawAffineConeAxes(bluntK);
    getKey();
  }
  
  // ======================================================================== //
  printStage("Convex pointed affine cone testing");
  auto convexPointedPredicate = [&](const Point2d& p) {
    return inside(p, convexPointedK);
  };
  auto convexPointedGroundTruth = [&](const Point2d& p) {
    return false;
  };
  sweepTest(convexPointedPredicate, convexPointedGroundTruth);
  if (TestParams::debug)
  {
    drawAffineConeAxes(convexPointedK);
    getKey();
  }

  // ======================================================================== //
  printStage("Blunt pointed affine cone testing");
  auto bluntPointedPredicate = [&](const Point2d& p) {
    return inside(p, bluntPointedK);
  };
  auto bluntPointedGroundTruth = [&](const Point2d& p) {
    return
      p.x() >= TestParams::w/2. && 
      p.y() == TestParams::h/2.;
  };
  sweepTest(bluntPointedPredicate, bluntPointedGroundTruth);
  if (TestParams::debug)
  {
    drawAffineConeAxes(bluntPointedK);
    getKey();
  }

  // ======================================================================== //
  printStage("Convex pointed affine cone testing: Case 2");
  auto convexPointedPredicate2 = [&](const Point2d& p) {
    return inside(p, convexPointedK2);
  };
  auto convexPointedGroundTruth2 = [&](const Point2d& p) {
    return false;
  };
  sweepTest(convexPointedPredicate2, convexPointedGroundTruth2);
  if (TestParams::debug)
  {
    drawAffineConeAxes(convexPointedK2);
    getKey();
  }

  // ======================================================================== //
  printStage("Blunt pointed affine cone testing: Case 2");
  auto bluntPointedPredicate2 = [&](const Point2d& p) {
    return inside(p, bluntPointedK2);
  };
  auto bluntPointedGroundTruth2 = [&](const Point2d& p) {
    return p.y() == TestParams::h/2.;
  };
  sweepTest(bluntPointedPredicate2, bluntPointedGroundTruth2);
  if (TestParams::debug)
  {
    drawAffineConeAxes(bluntPointedK2);
    getKey();
  }

}

TEST(DO_Geometry_Test, csgTest)
{
  Ellipse E(180, 100, 0., TestParams::center);
  AffineCone2 K(Point2d(1,0), Point2d(0,1), TestParams::center);

  CSG::Singleton<Ellipse> ell(E);
  CSG::Singleton<AffineCone2> cone(K);
  auto inter = ell*cone;

  int interArea = sweepCountPixels([&](const Point2d& p)
  { return inter.contains(p); });

  double trueArea = area(E)/4.;
  double relativeError = fabs(interArea -trueArea)/trueArea;
  EXPECT_NEAR(relativeError, 0, 1e-1);
  if (TestParams::debug)
  {
    CHECK(relativeError);
    getKey();
  }
}

TEST(DO_Geometry_Test, ellipseSectorArea)
{
  Ellipse E(180, 100, toRadian(30.), TestParams::center);

  CSG::Singleton<Ellipse> ell(E);
  Image<Rgb8> buffer(TestParams::w, TestParams::h);

  try {

    if (TestParams::debug)
    {
      if (!getActiveWindow())
        setAntialiasing(openWindow(TestParams::w, TestParams::h));
      clearWindow();
    }

    int steps = 18;
    ASSERT_EQ(steps%2, 0);
    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      //for (int i1 = i0+1; i1 <= i0+steps; ++i1)
      for (int i1 = 0; i1 < i0; ++i1) // revert order.
      {
        double theta1 = i1*2*M_PI/steps;
        double dTheta = fabs(theta1 - theta0);

        Triangle t(E.center(), E(theta0), E(theta1));

        CSG::Singleton<AffineCone2> cone(affineCone2(theta0+E.orientation(),
                                                     theta1+E.orientation(),
                                                     E.center()));
        auto E_and_Cone = ell*cone;
        auto E_minus_Cone = ell - E_and_Cone;
        auto inside_E_and_Cone = [&](const Point2d& p){ return E_and_Cone.contains(p); };
        auto inside_E_minus_Cone = [&](const Point2d& p){ return E_minus_Cone.contains(p); };

        int estimatedSectorArea = 0;
        //if (i1 - i0 < steps/2)
        if (i0-i1 > steps/2) // revert order
          estimatedSectorArea = sweepCountPixels(inside_E_and_Cone, buffer);
        else if (abs(i1-i0) == steps/2)
          estimatedSectorArea = area(E) / 2.;
        else
          estimatedSectorArea = sweepCountPixels(inside_E_minus_Cone, buffer);

        if (TestParams::debug)
        {
          /*cout << i0 << "     " << i1 << endl;
          cout << toDegree(theta0) << "     " << toDegree(theta1) << endl;
          cout << theta0 << "     " << theta1 << endl;*/
          display(buffer);
          //getKey();
        }

        double analyticSectorArea = sectorArea(E, theta0, theta1);
        double absError = fabs(estimatedSectorArea -analyticSectorArea);
        double relError = absError/estimatedSectorArea;
        if (estimatedSectorArea == 0 && absError < 1e-2)
          relError = absError;

        EXPECT_NEAR(relError, 0, 1e-1);
        if (TestParams::debug && relError > 1e-1)
        {
          printStage("Numerical error");
          cout << i0 << "     " << i1 << endl;
          CHECK(abs(i1-i0));
          CHECK(analyticSectorArea);
          CHECK(estimatedSectorArea);
          CHECK(absError);
          CHECK(relError);
          getKey();
        }
      }
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
    getKey();
  }
}

TEST(DO_Geometry_Test, ellipseSegmentArea)
{
  Ellipse E(180, 100, toRadian(42.), TestParams::center);
  Image<Rgb8> buffer(TestParams::w, TestParams::h);

  if (TestParams::debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  try
  {
    int steps = 18;
    ASSERT_EQ(steps%2, 0);

    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      for (int i1 = i0+1; i1 < i0+steps; ++i1)
      //for (int i1 = 0; i1 < i0; ++i1) // revert order.
      {
        double theta1 = i1*2*M_PI/steps;
        double dTheta = fabs(theta1 - theta0);

        const Point2d a(E(theta0));
        const Point2d b(E(theta1));
        Triangle t(E.center(), a, b);

        auto insideSegment = [&](const Point2d& p) -> bool
        { return (ccw(a,b,p) == -1) && inside(p, E); };

        double segArea1 = sweepCountPixels(insideSegment, buffer);
        double segArea2 = segmentArea(E, theta0, theta1);
        double sectArea = sectorArea(E, theta0, theta1);
        double triArea = area(t);

        display(buffer);
        drawTriangle(t, Green8, 3);


        double absError = fabs(segArea1 -segArea2);
        double relError = absError/segArea1;
        if (segArea1 == 0 && segArea2 < 1e-2)
          relError = absError;

        EXPECT_NEAR(relError, 0, 1e-1);
        if (TestParams::debug && relError > 2e-1)
        {
          printStage("Numerical error");
          CHECK(i0);
          CHECK(i1);
          CHECK(toDegree(theta0));
          CHECK(toDegree(theta1));
          CHECK(abs(i1-i0));
          CHECK(segArea1);
          CHECK(segArea2);
          CHECK(sectArea);
          CHECK(triArea);
          CHECK(relError);
          getKey();
        }
      }
    }
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
    getKey();
  }

}
#endif

void compareComputationTimes(const Ellipse& e1, const Ellipse& e2,
                             int discretization, double& bestSpeedGain)
{
  HighResTimer t;
  int times = 100;
  // Approximate computation.
  t.restart();
  double approxRatio, analyticRatio;
  for (int i = 0; i < times; ++i)
    approxRatio = approximateIntersectionUnionRatio(e1, e2, discretization);
  double approximateTime = t.elapsedMs()/double(times);
  
  t.restart();
  for (int i = 0; i < times; ++i)
    analyticRatio = analyticInterUnionRatio(e1, e2);
  double closedFormTime = t.elapsedMs()/double(times);

  double speedGain = approximateTime / closedFormTime;

  if (speedGain > bestSpeedGain)
  {
    bestSpeedGain = speedGain;

    printStage("Update speed gain");
    cout << "Approximate computation time = " << approximateTime << " ms" << endl;
    cout << "Closed-form computation time = " << closedFormTime << " ms" << endl;
    cout << "Best speed gain = " << speedGain << endl;
  }
}

void viewEllipses(const Ellipse& e1, const Ellipse& e2)
{
  // Verbose comments.
  /*cout << endl;
  cout << "Ellipse 1" << endl;
  cout << e1 << endl;
  cout << "Ellipse 2" << endl;
  cout << e2 << endl;*/
  
  if (!getActiveWindow())
    setAntialiasing(openWindow(TestParams::w, TestParams::h));
  
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

int countPixelInterArea(const Ellipse& e1, const Ellipse& e2)
{
  int interArea = 0;
  for (int y = -TestParams::h; y < TestParams::h; ++y)
  {
    for (int x = -TestParams::w; x < TestParams::w; ++x)
    {
      Point2d p(x,y);
      if (inside(p, e1) && inside(p, e2))
        ++interArea;
    }
  }
  return interArea;
}

TEST(DO_Geometry_Test, ellipseAlgorithmsTest)
{
  int discretization = 36;
  double bestSpeedGain = 0;
  long long attempts = 0;
  do
  {
    Ellipse e1(TestParams::randomEllipse(TestParams::w, TestParams::h));
    Ellipse e2(TestParams::randomEllipse(TestParams::w, TestParams::h));

    Point2d interPts[4];
    int numInter = computeEllipseIntersections(interPts, e1, e2);
    if (numInter != 2)
      continue;


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

    printStage("Intersection area computation");
    double exactInterArea = analyticInterArea(e1, e2);
    double approxInterArea = area(approxInter(e1, e2, discretization));
    double pixelInterArea = countPixelInterArea(e1, e2);

    CHECK(exactInterArea);
    CHECK(approxInterArea);
    CHECK(pixelInterArea);

    // ====================================================================== //
    // Speed comparison.
    // 
    // Experimental settings.
    bool compareSpeed = false;
    if (compareSpeed)
      compareComputationTimes(e1, e2, discretization, bestSpeedGain);

    // Summary
    double relativeError = abs(approxRatio-analyticRatio) / approxRatio;
    double absoluteError = abs(approxRatio-analyticRatio);

    double areaRatio = min(area(e1), area(e2)) / max(area(e1), area(e2));
    if (absoluteError > 0.05  /*&& areaRatio > 0.2*/)
    {
      viewEllipses(e1, e2);
      ostringstream oss;
      oss << "ATTEMPTS = " << attempts << " encountered computational issue";
      printStage(oss.str());

      CHECK(area(e1));
      CHECK(area(e2));
      CHECK(inside(e2.center(), e1));
      CHECK(inside(e1.center(), e2));
      CHECK(analyticInterArea(e1, e2, true));
      cout << "Approximate intersection-union area ratio = " << approxRatio << endl;
      cout << "Exact intersection-union area ratio = " << analyticRatio << endl;
      cout << "Relative error = " << relativeError << endl;
      cout << "Absolute error = " << absoluteError << endl;
      getKey();
    }

    ++attempts;
  } while(true);
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}