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