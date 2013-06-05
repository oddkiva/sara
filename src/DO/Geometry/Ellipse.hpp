/*
 * =============================================================================
 *
 *       Filename:  Ellipse.hpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  08/11/2010 19:24:00
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_GEOMETRY_ELLIPSE_HPP
#define DO_GEOMETRY_ELLIPSE_HPP

namespace DO {

  //! Ellipse class
	class Ellipse
	{
	public:
    Ellipse() {}
		Ellipse(double radius1, double radius2, double orientation, const Point2d& center)
		  : a_(radius1), b_(radius2), o_(orientation), c_(center) {}

		double r1() const { return a_; }
		double r2() const { return b_; }
		double o() const { return o_; }
		const Point2d& c() const { return c_; }

		double& r1() { return a_; }
		double& r2() { return b_; }
		double& o() { return o_; }
		Point2d& c() { return c_; }

    double area() const { return 3.14159265358979323846*a_*b_; }
    
    double F_(double theta) const
    {
      return a_*b_*0.5*atan(a_*tan(theta)/b_);
    }

    // Polar antiderivative
    double F(double theta) const
    {
      return a_*b_*0.5*
           ( theta 
           - atan( (b_-a_)*sin(2*theta) / ((b_+a_)+(b_-a_)*cos(2*theta)) ) );
      /*const double pi = 3.14159265358979323846;
      if (abs(theta) < pi/2.)
      {
        std::cout << "abs(theta) < pi/2." << std::endl;
        return F_(theta);
      }
      if (theta > 0)
      {
        std::cout << "pi/2. < theta < pi" << std::endl;
        return pi*a_*b_*0.5 - F_(pi-theta);
      }
      std::cout << "-pi < theta < -pi/2." << std::endl;
      return -pi*a_*b_*0.5 + F_(theta+pi);*/
    }

    bool isInside(const Point2d& p) const
    {
      return (p-c_).transpose()*shapeMat()*(p-c_) < 1.;
    }

    Matrix2d shapeMat() const;
    
    bool isWellDefined(double limit = 1e9) const
    { return (std::abs(r1()) < limit && std::abs(r2()) < limit); }

    void drawOnScreen(const Color3ub c, double scale = 1.) const;

	private:
		double a_, b_;
		double o_;
		Point2d c_;
	};

  //! I/O.
  std::ostream& operator<<(std::ostream& os, const Ellipse& e);

  //! Compute the ellipse from the conic equation
  Ellipse fromShapeMat(const Matrix2d& shapeMat, const Point2d& c);

  //! Compute the intersection union ratio approximately
  double approximateIntersectionUnionRatio(const Ellipse& e1, const Ellipse& e2,
                                           int n = 36,
                                           double limit = 1e9);

  //! Check polynomial solvers.
  //! TODO: Make a special library for polynomial solvers.
  void checkQuadraticEquationSolver();
  void checkCubicEquationSolver();
  void checkQuarticEquationSolver();

  void getEllipseIntersections(Point2d intersections[4], int& numInter,
                               const Ellipse& e1, const Ellipse& e2);

  int findFirstPoint(const Point2d pts[], int numPts);

  void ccwSortPoints(Point2d pts[], int numPts);

  double convexSectorArea(const Ellipse& e, const Point2d pts[]);

  double analyticInterUnionRatio(const Ellipse& e1, const Ellipse& e2);


} /* namespace DO */

#endif /* DO_GEOMETRY_ELLIPSE_HPP */