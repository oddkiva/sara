/*
 * =============================================================================
 *
 *       Filename:  BBox.hpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  24/03/2011 14:29:00
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_REGIONGROWING_BBOX_HPP
#define DO_REGIONGROWING_BBOX_HPP

namespace DO {

  struct BBox {
    Point2d topLeft, bottomRight;
    BBox() {}
    BBox(const Point2d& tl, const Point2d& br) : topLeft(tl), bottomRight(br) {}
    bool isInside(const Point2d& p) const;
    bool isDegenerate() const;
    bool invert();
    void drawOnScreen(const Color3ub& c, double scale = 1.) const;
    void print() const;
    double width() const { return std::abs(bottomRight.x() - topLeft.x()); }
    double height() const { return std::abs(bottomRight.y() - topLeft.y()); }

    Point2d tl() const { return topLeft; }
    Point2d tr() const { return topLeft+Point2d(width(), 0); }
    Point2d bl() const { return bottomRight-Point2d(width(), 0); }
    Point2d br() const { return bottomRight; }

    double x1() const { return  topLeft.x(); }
    double y1() const { return  topLeft.y(); }
    double x2() const { return  bottomRight.x(); }
    double y2() const { return  bottomRight.y(); }

    static BBox infBBox() {
      BBox b;
      b.topLeft.fill(-std::numeric_limits<double>::infinity());
      b.bottomRight.fill(std::numeric_limits<double>::infinity());
      return b;
    }

    static BBox nullBBox() {
        BBox b;
        b.topLeft.fill(-std::numeric_limits<double>::infinity());
        b.bottomRight.fill(-std::numeric_limits<double>::infinity());
        return b;
    }
  };

} /* namespace DO */

#endif /* DO_REGIONGROWING_BBOX_HPP */