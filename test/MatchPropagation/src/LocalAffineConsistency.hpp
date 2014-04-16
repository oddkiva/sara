// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#ifndef DO_GROWREGION_LOCALAFFINECONSISTENCY_HPP
#define DO_GROWREGION_LOCALAFFINECONSISTENCY_HPP

#include <DO/Features.hpp>
#include <DO/Match.hpp>
#include <DO/Geometry.hpp>

namespace DO {

  void getTriangleAnglesDegree(double angles[3], const Point2d t[3]);

  // Convenience function.
  Matrix3f affinityFromXToY(const Match& a, const Match& b, const Match& c);

  // Remember Jacobian $\mathbf{J} = (\nabla \mathbf{H})^T$.
  Matrix2d jacobian(const Matrix3d& H, const Vector2d& x);

  // It looks something is wrong here. Investigate...
  Matrix3d affinity(const Matrix3d& H, const Vector2d& x);

  // Helper function.
  Ellipse ellipseFromOERegion(const OERegion& f);

  OERegion transformOERegion(const OERegion& f, const Matrix3f& H);

  double computeOrientation(const Match& m, const Matrix3f& H);

  double angleDiffRadian(const OERegion& H_x, const OERegion& y);

  float centerDiff(const OERegion& H_x, const OERegion& y);

  // Actually more reliable but so much slower, also the times when the 
  // analytical computation of ellipses intersection is wrong (because of 
  // numerical issues) are VERY seldom (less than 1%).
  void compareOERegion(float& dist,
                       double& diff_angle_radian,
                       double& overlapRatio,
                       const OERegion& H_x, const OERegion& y,
                       bool approxEllInterArea = false);

} /*  namespace DO */

#endif /* DO_GROWREGION_LOCALAFFINECONSISTENCY_HPP */