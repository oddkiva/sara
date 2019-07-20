#pragma once

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>


using namespace DO::Sara;


class TestFixtureForPolygon
{
protected:
  int _width;
  int _height;
  double _a;
  double _b;
  Point2d _p1;
  Point2d _p2;
  Point2d _center;

public:
  TestFixtureForPolygon()
  {
    _width = 10;
    _height = 10;

    _a = 0.25*_width;
    _b = 0.75*_height;

    _p1 = Point2d(_a, _a);
    _p2 = Point2d(_b, _b);
    _center = Point2d(_width/2., _height/2.);
  }

  template <typename TestPred, typename GroundTruth>
  void sweep_check(const TestPred& pred, const GroundTruth& ground_truth)
  {
    for (int y = 0; y < _height; ++y)
    {
      for (int x = 0; x < _width; ++x)
      {
        Point2d p(x,y);
        BOOST_REQUIRE_EQUAL(ground_truth(p), pred(p));
      }
    }
  }

  template <typename TestPred>
  int sweep_count_pixels(const TestPred& pred)
  {
    int quantity = 0;
    for (int y = 0; y < _height; ++y)
    {
      for (int x = 0; x < _width; ++x)
      {
        Point2d p(x,y);
        if (pred(p))
          ++quantity;
      }
    }
    return quantity;
  }
};
