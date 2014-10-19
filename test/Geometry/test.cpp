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