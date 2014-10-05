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
#include <DO/Core.hpp>
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
               const GroundTruth& gdTruth,
               bool debug)
{
  if (debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  for (int y = -TestParams::h; y < TestParams::h; ++y)
  {
    for (int x = -TestParams::w; x < TestParams::w; ++x)
    {
      Point2d p(x,y);
      EXPECT_EQ(gdTruth(p), pred(p));

      if (pred(p) && gdTruth(p) && debug)
        drawPoint(x,y,Green8);
      if (pred(p) != gdTruth(p) && debug)
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
int sweepCountPixels(const TestPred& pred, bool debug)
{
  if (debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  int quantity = 0;
  for (int y = -TestParams::h; y < TestParams::h; ++y)
  {
    for (int x = -TestParams::w; x < TestParams::w; ++x)
    {
      Point2d p(x,y);
      if (pred(p))
      {
        ++quantity;
        if (debug)
          drawPoint(x,y, Green8);
      }
    }
  }

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

//#define RERUN
#ifdef RERUN
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
  sweepTest(predicate, groundTruth, TestParams::debug);


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

TEST(DO_Geometry_Test, quadTest)
{
  BBox bbox(TestParams::p1, TestParams::p2);
  Quad quad(bbox);

  EXPECT_NEAR(area(bbox), area(quad), 1e-10);

  auto predicate = [&](const Point2d& p) {
    return inside(p, quad);
  };
  auto groundTruth = [&](const Point2d& p) {
    return 
      TestParams::p1.x() <= p.x() && p.x() < TestParams::p2.x() &&
      TestParams::p1.y() <= p.y() && p.y() < TestParams::p2.y() ;
  };
  sweepTest(predicate, groundTruth, TestParams::debug);
}

TEST(DO_Geometry_Test, triangleTest)
{
  Triangle t1(Point2d(0,0), Point2d(100, 0), Point2d(100, 100));
  EXPECT_NEAR(area(t1), 1e4/2., 1e-10);

  Triangle t2(Point2d(100,0), Point2d(0, 0), Point2d(100, 100));
  EXPECT_NEAR(signedArea(t2), -1e4/2., 1e-10);

  Triangle t3(Point2d(50, 73), Point2d(350, 400), Point2d(25, 200));
  int pixelArea3 = sweepCountPixels(
    [&](Point2d& p){ return inside(p, t3); },
    TestParams::debug );
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
  sweepTest(convexPredicate, convexGroundTruth, TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(convexK);
    milliSleep(40);
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
  sweepTest(bluntPredicate, bluntGroundTruth, TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(bluntK);
    milliSleep(40);
  }
  
  // ======================================================================== //
  printStage("Convex pointed affine cone testing");
  auto convexPointedPredicate = [&](const Point2d& p) {
    return inside(p, convexPointedK);
  };
  auto convexPointedGroundTruth = [&](const Point2d& p) {
    return false;
  };
  sweepTest(convexPointedPredicate, convexPointedGroundTruth,
            TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(convexPointedK);
    milliSleep(40);
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
  sweepTest(bluntPointedPredicate, bluntPointedGroundTruth,
            TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(bluntPointedK);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Convex pointed affine cone testing: Case 2");
  auto convexPointedPredicate2 = [&](const Point2d& p) {
    return inside(p, convexPointedK2);
  };
  auto convexPointedGroundTruth2 = [&](const Point2d& p) {
    return false;
  };
  sweepTest(convexPointedPredicate2, convexPointedGroundTruth2,
            TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(convexPointedK2);
    milliSleep(40);
  }

  // ======================================================================== //
  printStage("Blunt pointed affine cone testing: Case 2");
  auto bluntPointedPredicate2 = [&](const Point2d& p) {
    return inside(p, bluntPointedK2);
  };
  auto bluntPointedGroundTruth2 = [&](const Point2d& p) {
    return p.y() == TestParams::h/2.;
  };
  sweepTest(bluntPointedPredicate2, bluntPointedGroundTruth2,
            TestParams::debug);
  if (TestParams::debug)
  {
    drawAffineCone(bluntPointedK2);
    milliSleep(40);
  }

}

TEST(DO_Geometry_Test, csgTest)
{
  Ellipse E(180, 100, 0., TestParams::center);
  AffineCone2 K(Point2d(1,0), Point2d(0,1), TestParams::center);

  CSG::Singleton<Ellipse> ell(E);
  CSG::Singleton<AffineCone2> cone(K);
  auto inter = ell*cone;

  int estimatedArea = sweepCountPixels(
    [&](const Point2d& p){ return inter.contains(p); },
    TestParams::debug );

  double trueArea = area(E)/4.;
  double relativeError = fabs(estimatedArea -trueArea)/trueArea;
  EXPECT_NEAR(relativeError, 0, 1e-1);
}

TEST(DO_Geometry_Test, ellipseSectorArea)
{
  Ellipse E(250, 150, toRadian(75.), TestParams::center);

  CSG::Singleton<Ellipse> ell(E);
  Image<Rgb8> buffer(TestParams::w, TestParams::h);

  if (TestParams::debug)
  {
    if (!getActiveWindow())
      setAntialiasing(openWindow(TestParams::w, TestParams::h));
    clearWindow();
  }

  try {

    int steps = 18;
    for (int i0 = 0; i0 <= steps; ++i0)
    {
      double theta0 = i0*2*M_PI/steps;
      for (int i1 = i0+1; i1 <= i0+steps; ++i1)
      {
        double theta1 = i1*2*M_PI/steps;
        
        // Compute the sector area in a closed form.
        double analyticSectorArea = sectorArea(E, theta0, theta1);

        // Build constructive solid geometry.
        CSG::Singleton<AffineCone2> cone (
          affineCone2(theta0+E.orientation(),
                      theta1+E.orientation(),
                      E.center()) );
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
        int estimatedSectorArea;
        if (i1 - i0 < steps/2)
          estimatedSectorArea = sweepCountPixels(inside_E_and_Cone, buffer);
        else if (abs(i1-i0) == steps/2)
          estimatedSectorArea = area(E) / 2.;
        else
          estimatedSectorArea = sweepCountPixels(inside_E_minus_Cone, buffer);

        // Absolute error and relative error.
        double absError = fabs(estimatedSectorArea -analyticSectorArea);
        double relError = absError/estimatedSectorArea;

        if (TestParams::debug)
          display(buffer);

        double thres = 1e-1;
        EXPECT_NEAR(relError, 0, thres);
        if (TestParams::debug && relError > thres)
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
  Ellipse E(270, 150, toRadian(42.), TestParams::center);
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

        auto insideSegment = [&](const Point2d& p) -> bool
        { return (ccw(a,b,p) == -1) && inside(p, E); };

        double segArea1 = sweepCountPixels(insideSegment, buffer);
        double segArea2 = segmentArea(E, theta0, theta1);

        if (TestParams::debug)
        {
          display(buffer);
          drawTriangle(t, Green8, 3);
        }

        double absError = fabs(segArea1 -segArea2);
        double relError = absError/segArea1;

        const double thres = 0.1;
        EXPECT_NEAR(relError, 0, thres);
        if (TestParams::debug && relError > thres)
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
                             int discretization, 
                             double& bestSpeedGain,
                             double& worstSpeedGain,
                             vector<double>& speedGainHistory)
{
  Timer t;
  int times = 100;
  // Approximate computation.
  t.restart();
  double approxRatio, analyticRatio;
  for (int i = 0; i < times; ++i)
    approxRatio = approxJaccardSimilarity(e1, e2, discretization);
  double approximateTime = t.elapsedMs()/double(times);
  
  t.restart();
  for (int i = 0; i < times; ++i)
    analyticRatio = analyticJaccardSimilarity(e1, e2);
  double closedFormTime = t.elapsedMs()/double(times);

  double speedGain = approximateTime / closedFormTime;
  speedGainHistory.push_back(speedGain);
  std::sort(begin(speedGainHistory), end(speedGainHistory));
  
  double meanSpeedGain = 0;
  meanSpeedGain = std::accumulate(begin(speedGainHistory), end(speedGainHistory), 0);
  meanSpeedGain /= speedGainHistory.size();

  if (speedGain > bestSpeedGain || speedGain < worstSpeedGain ||
      speedGainHistory.size() % 10000 == 0)
  {
    worstSpeedGain = min(speedGain, worstSpeedGain);
    bestSpeedGain = max(speedGain, bestSpeedGain);

    printStage("Update speed gain");
    cout << "Approximate computation time = " << approximateTime << " ms" << endl;
    cout << "Closed-form computation time = " << closedFormTime << " ms" << endl;
    cout << "Worst speed gain = " << worstSpeedGain << endl;
    cout << "Best speed gain = " << bestSpeedGain << endl;
    cout << "Mean speed gain = " << meanSpeedGain << endl;
    cout << "[0.7]-ian speed gain = " << speedGainHistory[0.7*speedGainHistory.size()] << endl;
  }
}

void viewEllipses(const Ellipse& e1, const Ellipse& e2)
{ 
  if (!getActiveWindow())
    setAntialiasing(openWindow(TestParams::w, TestParams::h));
  
  clearWindow();
  drawEllipse(e1, Red8, 3);
  drawEllipse(e2, Blue8, 3);
  
  Point2d interPts[4];
  int numInter = computeIntersectionPoints(interPts, e1, e2);
  cout << "\nIntersection count = " << numInter << endl;
  for (int i = 0; i < numInter; ++i)
  {
    fillCircle(interPts[i].cast<float>(), 5.f, Green8);
    drawString(interPts[i].x()+10, interPts[i].y()+10, toString(i), Green8);
    cout << "[" << i << "] " << interPts[i].transpose() << endl;
  }
}

TEST(DO_Geometry_Test, ellipseAlgorithmsTest)
{
  bool debug = false;
  int discretization = 6;
  double worstSpeedGain = std::numeric_limits<double>::max();
  double bestSpeedGain = 0;
  vector<double> speedGainHistory;
  long long attempts = 0;

  do
  {
    Ellipse e1(TestParams::randomEllipse(TestParams::w, TestParams::h));
    Ellipse e2(TestParams::randomEllipse(TestParams::w, TestParams::h));

    Point2d interPts[4];
    int numInter = computeIntersectionPoints(interPts, e1, e2);

    // Visualize.
    if (debug)
      viewEllipses(e1, e2);

    // ====================================================================== //
    // Approximate computation of intersecting ellipses.
    //
    // Very reliable and based on Sutherland-Hodgman clipping method.
    // Approximating an ellipse by an 8-sided polygon turns out to be actually 
    // a quite good approximation in practice!
    // We use in our method.
    //
    // For very precise computation of the Jaccard similarity, a 36-sided 
    // polygon is a good choice.
    double approxRatio = approxJaccardSimilarity(e1, e2, discretization);

    // ====================================================================== //
    // Closed-form computation.
    // The computation is really good but there are some numerical precision
    // issues to investigate. For now, numerical accuracy is tuned by hand and
    // numerical error occur, they look acceptable.
    //
    // We need to solve accuracy in the intersection points of two overlapping 
    // ellipses...
    //
    // In this test, the computation is wrong every 600,000 attempts...
    double analyticRatio = analyticJaccardSimilarity(e1, e2);

    if (debug)
    {
      printStage("Intersection area computation");
      double exactInterArea = analyticIntersection(e1, e2);
      double approxInterArea = area(approxIntersection(e1, e2, discretization));
      auto insideEllipseIntersection = [&](const Point2d& p) {
        return inside(p, e1) && inside(p, e2); 
      };
      double pixelInterArea = sweepCountPixels(insideEllipseIntersection, false);
      CHECK(exactInterArea);
      CHECK(approxInterArea);
      CHECK(pixelInterArea);
    }

    // ====================================================================== //
    // Speed comparison.
    // 
    // Experimental settings.
    bool compareSpeed = false;
    if (compareSpeed)
      compareComputationTimes(e1, e2, discretization, 
                              bestSpeedGain, worstSpeedGain, 
                              speedGainHistory);

    // Summary
    double jaccardRelError = abs(approxRatio-analyticRatio) / approxRatio;
    double jaccardAbsError = abs(approxRatio-analyticRatio);

    bool degen1 = min(e1.radius1(), e1.radius2())/max(e1.radius1(), e1.radius2()) < 1e-2;
    bool degen2 = min(e2.radius1(), e2.radius2())/max(e2.radius1(), e2.radius2()) < 1e-2;

    if (jaccardAbsError > 1e-1 && (!degen1 && !degen2))
    {
      viewEllipses(e1, e2);
      ostringstream oss;
      oss << "ATTEMPTS = " << attempts << " encountered computational issue";
      printStage(oss.str());

      CHECK(e1.radius1());
      CHECK(e1.radius2());
      CHECK(e2.radius1());
      CHECK(e2.radius2());
      CHECK(area(e1));
      CHECK(area(e2));
      CHECK(inside(e2.center(), e1));
      CHECK(inside(e1.center(), e2));
      CHECK(analyticIntersection(e1, e2));
      cout << "Approximate intersection-union area ratio = " << approxRatio << endl;
      cout << "Exact intersection-union area ratio = " << analyticRatio << endl;
      cout << "Relative error = " << jaccardRelError << endl;
      cout << "Absolute error = " << jaccardAbsError << endl;
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