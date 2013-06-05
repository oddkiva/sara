/*
 * =============================================================================
 *
 *       Filename:  Quad.hpp
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

#ifndef DO_GEOMETRY_QUAD_HPP
#define DO_GEOMETRY_QUAD_HPP

namespace DO {

  struct Quad {
    Point2d a, b, c, d; // Important: must be enumerated in ccw order.
    Vector3d lineEqns[4];
    Quad() {}
    Quad(const BBox& bbox);
    Quad(const Point2d& a_, const Point2d& b_, const Point2d& c_, const Point2d& d_);

    void dilate(double step);
    void applyH(const Matrix3d& H);
    bool invertEnumerationOrder();

    BBox bbox() const;
    bool isAlmostSimilar(const Quad& quad) const;

    bool isInside(const Point2d& p) const;
    bool intersect(const Quad& quad) const;
    double overlap(const Quad& quad) const;
    
    void print() const;
    void drawOnScreen(const Color3ub& c, double scale = 1.) const;
  };

  bool readQuads(std::vector<Quad>& quads, const std::string& filePath);
  bool writeQuads(const std::vector<Quad>& quads, const std::string& filePath);

} /* namespace DO */

#endif /* DO_GEOMETRY_QUAD_HPP */