// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SARA_GEOMETRY_BBOX_HPP
#define DO_SARA_GEOMETRY_BBOX_HPP


#include <stdexcept>
#include <vector>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO {

  class BBox
  {
  public:
    //! Default constructor.
    BBox() = default;

    //! \brief Constructor from the BBox end points.
    BBox(const Point2d& top_left, const Point2d& bottom_right)
      : _top_left(top_left), _bottom_right(bottom_right)
    {
      if ( _top_left.x() > _bottom_right.x() ||
           _top_left.y() > _bottom_right.y() )
      {
        const char *msg = "Top-left and bottom-right corners are wrong!";
        throw std::logic_error(msg);
      }
    }

    //! @{
    //! \brief Constructor from a point set.
    BBox(const Point2d *begin, const Point2d *end)
    {
      if (!begin)
      {
        const char *msg = "The array of points seems wrong.";
        throw std::logic_error(msg);
      }
      _top_left = _bottom_right = *begin;
      for (const Point2d *p = begin; p != end; ++p)
      {
        _top_left.x() = std::min(_top_left.x(), p->x());
        _top_left.y() = std::min(_top_left.y(), p->y());
        _bottom_right.x() = std::max(_bottom_right.x(), p->x());
        _bottom_right.y() = std::max(_bottom_right.y(), p->y());
      }
    }

    BBox(const std::vector<Point2d>& points)
      : BBox(&points.front(), &points.front() + points.size())
    {
    }
    //! @}

    //! @{
    //! \brief Return BBox vertex.
    Point2d& top_left() { return _top_left; }
    Point2d& bottom_right() { return _bottom_right; }

    const Point2d& top_left() const { return _top_left; }
    const Point2d& bottom_right() const { return _bottom_right; }
    Point2d top_right() const { return _top_left + Point2d(width(), 0); }
    Point2d bottom_left() const { return _bottom_right - Point2d(width(), 0); }
    //! @}

    //! @{
    //! \brief Return BBox coordinates.
    double& x1() { return  _top_left.x(); }
    double& y1() { return  _top_left.y(); }
    double& x2() { return  _bottom_right.x(); }
    double& y2() { return  _bottom_right.y(); }

    double x1() const { return  _top_left.x(); }
    double y1() const { return  _top_left.y(); }
    double x2() const { return  _bottom_right.x(); }
    double y2() const { return  _bottom_right.y(); }
    //! @}

    //! @{
    //! \brief Return BBox sizes.
    double width() const  { return std::abs(_bottom_right.x() - _top_left.x()); }
    double height() const { return std::abs(_bottom_right.y() - _top_left.y()); }
    Vector2d sizes() const { return _bottom_right - _top_left; }
    //! @}

    //! \brief Return BBox center.
    Point2d center() const { return 0.5*(_top_left + _bottom_right); }

    //! @{
    //! \brief Convenience functions.
    static BBox infinite()
    {
      BBox b;
      b.top_left().fill(-std::numeric_limits<double>::infinity());
      b.bottom_right().fill(std::numeric_limits<double>::infinity());
      return b;
    }

    static BBox zero()
    {
      BBox b(Point2d::Zero(), Point2d::Zero());
      return b;
    }
    //! @}

  private:
    //! @{
    Point2d _top_left;
    Point2d _bottom_right;
    //! @}

  };

  //! @{
  //! Utility functions.
  double area(const BBox& bbox);
  bool inside(const Point2d& p, const BBox& bbox);
  bool degenerate(const BBox& bbox, double eps = 1e-3);
  bool intersect(const BBox& bbox1, const BBox& bbox2);
  double jaccard_similarity(const BBox& bbox1, const BBox& bbox2);
  double jaccard_distance(const BBox& bbox1, const BBox& bbox2);
  //! @}

  //! I/O.
  std::ostream& operator<<(std::ostream& os, const BBox& bbox);

  //! Return the intersection of two BBoxes.
  BBox intersection(const BBox& bbox1, const BBox& bbox2);


} /* namespace DO */

#endif /* DO_SARA_GEOMETRY_BBOX_HPP */
