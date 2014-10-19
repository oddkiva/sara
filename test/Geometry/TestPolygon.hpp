#ifndef TEST_POLYGON_HPP
#define TEST_POLYGON_HPP

#include <gtest/gtest.h>

#include <DO/Core/EigenExtension.hpp>


using namespace DO;


class TestPolygon : public testing::Test
{
protected:
  int w;
  int h;
  double a;
  double b;
  Point2d p1;
  Point2d p2;
  Point2d center;

  TestPolygon()
  {
    w = 512;
    h = 512;

    a = 0.25*w;
    b = 0.75*h;

    p1 = Point2d(a, a);
    p2 = Point2d(b, b);
    center = Point2d(w/2., h/2.);
  }

  template <typename TestPred, typename GroundTruth>
  void sweep_check(const TestPred& pred, const GroundTruth& ground_truth)
  {
    for (int y = -h; y < h; ++y)
    {
      for (int x = -w; x < w; ++x)
      {
        Point2d p(x,y);
        EXPECT_EQ(ground_truth(p), pred(p));
      }
    }
  }

  template <typename TestPred>
  int sweep_count_pixels(const TestPred& pred)
  {
    int quantity = 0;
    for (int y = -h; y < h; ++y)
    {
      for (int x = -w; x < w; ++x)
      {
        Point2d p(x,y);
        if (pred(p))
          ++quantity;
      }
    }
    return quantity;
  }
};


#endif /* TEST_POLYGON_HPP */