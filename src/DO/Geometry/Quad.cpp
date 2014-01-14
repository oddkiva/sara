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

#pragma warning (disable : 4267 4503)

#include <DO/Geometry.hpp>
#include <DO/Graphics.hpp>
#include <iostream>
#include <fstream>

using namespace std;

namespace DO {

  Vector3d computeLineEqn(const Point2d& a, const Point2d& b)
  {
    // Direction vector
    Vector2d u(b - a);
    u.normalize();
    //Inward normal
    Vector2d n(u[1], -u[0]);
    return Vector3d(n[0], n[1],-n[0]*a[0]-n[1]*a[1]);
  }

  Quad::Quad(const BBox& bbox)
    : a(bbox.topLeft())
    , b(bbox.topRight())
    , c(bbox.bottomRight())
    , d(bbox.bottomLeft())
  {
    lineEqns[0] = computeLineEqn(a, b);
    lineEqns[1] = computeLineEqn(b, c);
    lineEqns[2] = computeLineEqn(c, d);
    lineEqns[3] = computeLineEqn(d, a);
  }

  Quad::Quad(const Point2d& a_, const Point2d& b_, const Point2d& c_, const Point2d& d_)
    : a(a_), b(b_), c(c_), d(d_)
  {
    lineEqns[0] = computeLineEqn(a, b);
    lineEqns[1] = computeLineEqn(b, c);
    lineEqns[2] = computeLineEqn(c, d);
    lineEqns[3] = computeLineEqn(d, a);
  }

  void Quad::applyH(const Matrix3d& H)
  {
    Vector3d ha = H*(Vector3d() << a, 1.).finished(); ha/=ha(2);
    Vector3d hb = H*(Vector3d() << b, 1.).finished(); hb/=hb(2);
    Vector3d hc = H*(Vector3d() << c, 1.).finished(); hc/=hc(2);
    Vector3d hd = H*(Vector3d() << d, 1.).finished(); hd/=hd(2);

    a = ha.head<2>();
    b = hb.head<2>();
    c = hc.head<2>();
    d = hd.head<2>();

    lineEqns[0] = computeLineEqn(a, b);
    lineEqns[1] = computeLineEqn(b, c);
    lineEqns[2] = computeLineEqn(c, d);
    lineEqns[3] = computeLineEqn(d, a);
  }

  void Quad::dilate(double step)
  {
    Point2d center;
    center = (a+b+c+d)/4.;
    a += (a-center)/(a-center).norm() * step;
    b += (b-center)/(b-center).norm() * step;
    c += (c-center)/(c-center).norm() * step;
    d += (d-center)/(d-center).norm() * step;
  }

  bool Quad::invertEnumerationOrder()
  {
    Point2d center;
    center = (a+b+c+d)/4.;
    if (!isInside(center))
    {
      std::swap(b,c);
      std::swap(d,a);
      lineEqns[0] = computeLineEqn(a, b);
      lineEqns[1] = computeLineEqn(b, c);
      lineEqns[2] = computeLineEqn(c, d);
      lineEqns[3] = computeLineEqn(d, a);
      return true;
    }
    if (!isInside(center))
      cerr << "Error: please check that the quad." << endl;
    return false;
  }

  BBox Quad::bbox() const
  {
    Point2d pts[4] = { a, b, c, d };
    Point2d tl(a), br(a);
    for (int i = 0; i < 4; ++i)
    {
      if (tl.x() > pts[i].x())
        tl.x() = pts[i].x();
      if (tl.y() > pts[i].y())
        tl.y() = pts[i].y();

      if (br.x() < pts[i].x())
        br.x() = pts[i].x();
      if (br.y() < pts[i].y())
        br.y() = pts[i].y();
    }
    
    return BBox(tl, br);
  }

  bool Quad::isAlmostSimilar(const Quad& quad) const
  {
    double thres = 1e-3;
    double thres2 = thres*thres;
    double da, db, dc, dd;
    da = (a-quad.a).squaredNorm();
    db = (b-quad.b).squaredNorm();
    dc = (c-quad.c).squaredNorm();
    dd = (d-quad.d).squaredNorm();
    return (da < thres2 && db < thres2 && dc < thres2 && dd < thres2);
  }

  bool Quad::isInside(const Point2d& p) const
  {
    double signs[4];
    for (int i = 0; i < 4; ++i)
      signs[i] = lineEqns[i].dot( (Vector3d() << p, 1.).finished() );

    for (int i = 0; i < 4; ++i)
      if(signs[i] > 0)
        return false;
    return true;
  }

  bool Quad::intersect(const Quad& quad) const
  {
    return true;
  }

  double Quad::overlap(const Quad& quad) const
  {
    if (isAlmostSimilar(quad))
      return 1.;

    return 0.;  
  }

  void Quad::print() const
  {
    cout << "a = " << a.transpose() << endl;
    cout << "b = " << b.transpose() << endl;
    cout << "c = " << c.transpose() << endl;
    cout << "d = " << d.transpose() << endl;
  }

  void Quad::drawOnScreen(const Color3ub& col, double z) const
  {
    Point2d az(a*z);
    Point2d bz(b*z);
    Point2d cz(c*z);
    Point2d dz(d*z);
    drawLine(az, bz, col);
    drawLine(bz, cz, col);
    drawLine(cz, dz, col);
    drawLine(dz, az, col);
  }

  bool readQuads(vector<Quad>& quads, const string& filePath)
  {
    ifstream f(filePath.c_str());
    if (!f.is_open()) {
      std::cerr << "Cant open file " << filePath << std::endl;    
      return false;
    }

    int n;
    f >> n;
    quads.clear();
    quads.resize(n);  

    for (int i = 0; i < n; ++i)
    {
      Point2d a, b, c, d;
      f >> a >> b >> c >> d;
      quads[i] = Quad(a,b,c,d);
    }
    f.close();

    return true;
  }

  bool writeQuads(const vector<Quad>& quads, const string& filePath)
  {
    ofstream f(filePath.c_str());
    if (!f.is_open()) {
      std::cerr << "Cant open file " << filePath << std::endl;    
      return false;
    }

    f << quads.size() << endl;
    for (vector<Quad>::const_iterator q = quads.begin(); q != quads.end(); ++q)
      f << q->a.transpose() << " " << q->b.transpose() << " " << q->c.transpose() << " " << q->d.transpose() << endl;
    f.close();

    return true;
  }

} /* namespace DO */