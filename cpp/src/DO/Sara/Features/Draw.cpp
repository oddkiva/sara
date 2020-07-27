// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/Graphics.hpp>

using namespace std;


namespace DO { namespace Sara {

  void OERegion::draw(const Color3ub& color, float scale,
                      const Point2f& offset) const
  {
    const auto& z = scale;

    // Solve characteristic equation to find eigenvalues
    auto svd = JacobiSVD<Matrix2f>{shape_matrix, ComputeFullU};
    const Vector2f& D = svd.singularValues();
    const Matrix2f& U = svd.matrixU();

    // Eigenvalues are l1 and l2.
    const Vector2f radii{D.cwiseSqrt().cwiseInverse()};

    // Caveat: the shape matrix is actually the scale matrix up to a factor
    // sqrt(2).
    //
    // In slides:
    //   http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
    // the blob radius is the scale multiplied sqrt(2).
    constexpr auto sqrt_two = static_cast<float>(M_SQRT2);
    const auto a = radii(0) * sqrt_two;
    const auto b = radii(1) * sqrt_two;

    // Orientation.
    const auto ori_degree = atan2(U(1, 0), U(0, 0)) *  //
                            180 / static_cast<float>(M_PI);

    // Start and end points of orientation line.
    const Matrix2f& L = affinity().block(0, 0, 2, 2);
    const Vector2f& p1 = z * (center() + offset);
    const Vector2f& p2 = p1 + z * sqrt_two * L * Vector2f::UnitX();

    // Draw.
    if (z * a > 1.f && z * b > 1.f && (p1 - p2).squaredNorm() > 1.f)
    {
      // Contour of orientation line.
      draw_line(p1, p2, Black8, 5);
      // Contour of ellipse.
      draw_ellipse(p1, z * a, z * b, ori_degree, Black8, 5);
      // Fill-in of orientation line.
      draw_line(p1, p2, color, 3);
      // Fill-in of ellipse.
      draw_ellipse(p1, z * a, z * b, ori_degree, color, 3);
    }
    else
    {
      const auto& z = scale;
      const auto cross_offset = 3.0f;

      const Vector2f& p1 = z * (center() + offset);
      const Vector2f& c1 = p1 - cross_offset * Vector2f::UnitX();
      const Vector2f& c2 = p1 + cross_offset * Vector2f::UnitX();
      const Vector2f& c3 = p1 - cross_offset * Vector2f::UnitY();
      const Vector2f& c4 = p1 + cross_offset * Vector2f::UnitY();

      draw_line(c1, c2, Black8, 5);
      draw_line(c3, c4, Black8, 5);
      draw_line(c1, c2, color, 3);
      draw_line(c3, c4, color, 3);
    }
  }

}}  // namespace DO::Sara
