/*
 * =============================================================================
 *
 *       Filename:  BBox.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  24/03/2011 14:34:00
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>

using namespace std;

namespace DO {

  bool BBox::isInside(const Point2d& p) const
  {
    return (
      p.x() >= topLeft.x() && p.x() <= bottomRight.x() &&
      p.y() >= topLeft.y() && p.y() <= bottomRight.y() 
      );
  }

	bool BBox::isDegenerate() const
	{
		bool degenerate = (
      topLeft.x() == bottomRight.x() || 
      topLeft.y() == bottomRight.y()
      );
		if (degenerate)
			std::cout << "degenerate" << std::endl;

		return degenerate;
	}

	bool BBox::invert()
	{
    bool inverted = false;

		if (topLeft.x() > bottomRight.x())
    {
      std::swap(topLeft.x(), bottomRight.x());
      inverted = true;
    }
    if (topLeft.y() > bottomRight.y())
    {
      std::swap(topLeft.y(), bottomRight.y());
      inverted = true;
    }
    return inverted;
	}

  void BBox::print() const
  {
    cout << "top left " << topLeft.transpose() << endl;
    cout << "bottom right " << bottomRight.transpose() << endl;
  }

  void BBox::drawOnScreen(const Color3ub& col, double z) const
  {
    Point2d tl(z*topLeft);
    Point2d br(z*bottomRight);
    Point2d sz(br-tl);
    drawRect(tl.cast<int>()(0), tl.cast<int>()(1), sz.cast<int>()(0), sz.cast<int>()(1), col);
  }

} /* namespace DO */