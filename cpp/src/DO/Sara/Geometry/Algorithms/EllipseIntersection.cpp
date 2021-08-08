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

#include <array>
#include <iostream>
#include <vector>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Math/NewtonRaphson.hpp>

#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>
#include <DO/Sara/Geometry/Algorithms/EllipseIntersection.hpp>
#include <DO/Sara/Geometry/Algorithms/SutherlandHodgman.hpp>
#include <DO/Sara/Geometry/Tools/PolynomialRoots.hpp>
#include <DO/Sara/Geometry/Tools/Utilities.hpp>


namespace DO { namespace Sara {

  // ======================================================================== //
  // Computation of intersecting points of intersecting ellipses
  // ======================================================================== //
  using ConicEquation = std::array<double, 6>;

  void print_conic_equation(const ConicEquation& s)
  {
    std::cout << s[0] << " + " << s[1] << " x + " << s[2] << " y + " << s[3]
              << " x^2 + " << s[4] << " xy + " << s[5] << " y^2 = 0"
              << std::endl;
  }

  auto conic_equation(const Ellipse& e) -> ConicEquation
  {
    auto s = ConicEquation{};
    
    const auto M = shape_matrix(e);

    s[0] = e.center().x() * M(0, 0) * e.center().x() +
           2 * e.center().x() * M(0, 1) * e.center().y() +
           e.center().y() * M(1, 1) * e.center().y() - 1.0;
    s[1] = -2.0 * (M(0, 0) * e.center().x() + M(0, 1) * e.center().y());
    s[2] = -2.0 * (M(1, 0) * e.center().x() + M(1, 1) * e.center().y());
    s[3] = M(0, 0);
    s[4] = 2.0 * M(0, 1);
    s[5] = M(1, 1);

    return s;
  }

  auto quartic_equation(const ConicEquation& s, const ConicEquation& t)
      -> Polynomial<double, 4>
  {
    double d[6][6];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        d[i][j] = s[i] * t[j] - s[j] * t[i];

    Polynomial<double, 4> u;
    u[0] = d[3][1] * d[1][0] - d[3][0] * d[3][0];
    u[1] = d[3][4] * d[1][0] + d[3][1] * (d[4][0] + d[1][2]) -
           2 * d[3][2] * d[3][0];
    u[2] = d[3][4] * (d[4][0] + d[1][2]) + d[3][1] * (d[4][2] - d[5][1]) -
           d[3][2] * d[3][2] - 2 * d[3][5] * d[3][0];
    u[3] = d[3][4] * (d[4][2] - d[5][1]) + d[3][1] * d[4][5] -
           2 * d[3][5] * d[3][2];
    u[4] = d[3][4] * d[4][5] - d[3][5] * d[3][5];
    return u;
  }

  auto sigma_polynomial(const ConicEquation& s, double y)
      -> Polynomial<double, 2>
  {
    auto sigma = Polynomial<double, 2>{};
    sigma[0] = s[0] + s[2] * y + s[5] * y * y;
    sigma[1] = s[1] + s[4] * y;
    sigma[2] = s[3];
    return sigma;
  }

  inline auto evaluate_conic_equation_at(double x, double y,
                                         const ConicEquation& s) -> double
  {
    return s[0] + s[1] * x + s[2] * y + s[3] * x * x + s[4] * x * y +
           s[5] * y * y;
  }

  auto add_intersection_point(const ConicEquation& s, const ConicEquation& t,
                              double y,  //
                              std::vector<Point2d>& intersections)
      -> void
  {
    const auto sigma = sigma_polynomial(s, y);
    const auto tau = sigma_polynomial(t, y);
    const auto x_denom = sigma[1] * tau[2] - sigma[2] * tau[1];

    auto intersection_point_is_in_both_ellipse = [&s, &t](double x, double y) {
      const auto e1_at_x1y = evaluate_conic_equation_at(x, y, s);
      const auto e2_at_x1y = evaluate_conic_equation_at(x, y, t);
      return std::abs(e1_at_x1y) < 1e-2 && std::abs(e2_at_x1y) < 1e-2;
    };

    if (x_denom == 0)
    {
      auto x1s = std::complex<double>{};
      auto x2s = std::complex<double>{};
      auto real_root_s = false;
      roots(sigma, x1s, x2s, real_root_s);

      if (!real_root_s)
        return;

      const auto x1 = std::real(x1s);
      const auto x2 = std::real(x2s);

      if (intersection_point_is_in_both_ellipse(x1, y))
        intersections.emplace_back(x1, y);

      if (intersection_point_is_in_both_ellipse(x2, y))
        intersections.emplace_back(x2, y);
    }
    else
    {
      const auto x_num = sigma[2] * tau[0] - sigma[0] * tau[2];
      const auto x = x_num / x_denom;
      if (intersection_point_is_in_both_ellipse(x, y))
        intersections.emplace_back(x, y);
    }
  }

  auto compute_intersection_points(const Ellipse& e1, const Ellipse& e2)
      -> std::vector<Point2d>
  {
    // Rescale ellipse to try to improve numerical accuracy.
    Point2d center;
    center = 0.5 * (e1.center() + e2.center());

    // Center the ellipse.
    // TODO: rescale the ellipses.
    Ellipse ee1(e1), ee2(e2);
    ee1.center() -= center;
    ee2.center() -= center;

    // Form the conic equations.
    const auto s = conic_equation(ee1);
    const auto t = conic_equation(ee2);

    // Extract the quartic equations from the conic equations.
    auto u = quartic_equation(s, t);
    // Rescale the polynomial.
    for (auto i = 0; i < 4; ++i)
      u[i] /= u[4];
    u[4] = 1;

    // Find the roots using the analytic form.
    auto y = std::array<std::complex<double>, 4>{};
    roots(u, y[0], y[1], y[2], y[3]);

    // Keep the real roots.
    auto y_polished = std::vector<double>{};
    for (auto i = 0; i < 4; ++i)
    {
      // We should be close to the real solution if we are dealing with an
      // actual intersection point.
      using std::imag;
      using std::real;

      const auto is_y_real = std::abs(imag(y[i])) < 1e-2 * std::abs(real(y[i]));
      if (!is_y_real)
        continue;

      y_polished.push_back(std::real(y[i]));
    }

    constexpr auto eps = 1e-4;
    constexpr auto squared_eps = eps * eps;
    std::sort(y_polished.begin(), y_polished.end());
    const auto y_last =
        std::unique(y_polished.begin(), y_polished.end(),
                    [&eps](const double& a, const auto& b) {
                      return std::abs(a - b) < eps;
                    });
    y_polished.resize(y_last - y_polished.begin());

#define POLISH_ROOTS
#ifdef POLISH_ROOTS
    // Now polish the roots with Newton-Raphson optimization method.
    for (auto& y_polished_i : y_polished)
    {
      auto P = Univariate::UnivariatePolynomial<double>{4};
      for (auto d = 0; d <= u.Degree; ++d)
        P[d] = u[d];

      auto root_polisher = Univariate::NewtonRaphson{P};
      y_polished_i = root_polisher(y_polished_i, 10);
    }
#endif

    // Finally form the intersection points for which the y-coordinate is real.
    auto intersections = std::vector<Point2d>{};
    for (const auto& y_polished_i: y_polished)
      add_intersection_point(s, t, y_polished_i, intersections);
 
    // Remove any duplicates.
    const auto intersection_last =
        std::unique(intersections.begin(), intersections.end(),
                    [&squared_eps](const Point2d& a, const Point2d& b) {
                      return (a - b).squaredNorm() < squared_eps;
                    });

    const auto num_intersections =
        static_cast<int>(intersection_last - intersections.begin());
    if (num_intersections > 4)
      throw std::runtime_error{
          "Ellipse intersections: too many intersections!"};
    intersections.resize(num_intersections);

    return intersections;
  }

  void orientation(double* ori, const Point2d* pts, int numPoints,
                   const Ellipse& e)
  {
    const Vector2d u(unit_vector2(e.orientation()));
    const Vector2d v(-u(1), u(0));
    std::transform(pts, pts + numPoints, ori, [&](const Point2d& p) -> double {
      const Vector2d d(p - e.center());
      return atan2(v.dot(d), u.dot(d));
    });
  }


  // ======================================================================== //
  // Computation of the area of intersecting ellipses.
  // ======================================================================== //
  double analytic_intersection_area(const Ellipse& E_0, const Ellipse& E_1)
  {
    // The ellipse intersection is not entirely robust numerically. I will
    // investigate later.
    // 
    // For now I prefer to have everything working more or
    // less OK because of the match propagation module.
#ifndef DEBUG_ELLIPSE_INTERSECTION_IMPLEMENTATION
    auto inter_pts = compute_intersection_points(E_0, E_1);
#else
      SARA_CHECK(numInter);
#ifdef DEBUG_ELLIPSE_INTERSECTION_IMPLEMENTATION
      if (!active_window())
        set_antialiasing(create_window(512, 512));
      clear_window();
      draw_ellipse(E_0, Red8, 3);
      draw_ellipse(E_1, Blue8, 3);

      Quad Q_0(oriented_bbox(E_0));
      Quad Q_1(oriented_bbox(E_1));

      BBox b0(&Q_0[0], &Q_0[0] + 4);
      BBox b1(&Q_1[0], &Q_1[0] + 4);
      b0.top_left() = b0.top_left().cwiseMin(b1.top_left());
      b0.bottom_right() = b0.bottom_right().cwiseMax(b1.bottom_right());

      draw_quad(Q_0, Red8, 3);
      draw_quad(Q_1, Blue8, 3);
      draw_bbox(b0, Green8, 3);
      get_key();
#endif

      // now rescale the ellipse.
      Point2d center(b0.center());
      Vector2d delta(b0.sizes() - center);
      delta = delta.cwiseAbs();

      Ellipse EE_0, EE_1;
      Matrix2d S_0 =
          delta.asDiagonal() * shape_matrix(E_0) * delta.asDiagonal();
      Matrix2d S_1 =
          delta.asDiagonal() * shape_matrix(E_1) * delta.asDiagonal();

      SARA_CHECK(shape_matrix(E_0));
      SARA_CHECK(S_0);

      Vector2d c_0 = E_0.center() - center;
      Vector2d c_1 = E_1.center() - center;
      EE_0 = construct_from_shape_matrix(S_0, c_0);
      EE_1 = construct_from_shape_matrix(S_1, c_1);
      int num_inter = compute_intersection_points(inter_pts, EE_0, EE_1);

      for (int i = 0; i < num_inter; ++i)
        inter_pts[i] = delta.asDiagonal() * inter_pts[i] + center;
    }
#endif

    if (inter_pts.size() > 2)
    {
      Detail::PtCotg work[4];
      Detail::sort_points_by_polar_angle(&inter_pts[0], work, inter_pts.size());
    }

    // SPECIAL CASE.
    //
    // If there is at most one intersection point, then either one of the
    // ellipse is included in the other.
    if (inter_pts.size() < 2)
    {
      if (E_0.contains(E_1.center()) || E_1.contains(E_0.center()))
        return std::min(area(E_0), area(E_1));
    }

    // GENERIC CASE
    //
    // Compute the relative orientation of the intersection points w.r.t. the
    // ellipse orientation.
    double o_0[4];
    double o_1[4];
    const auto num_intersections = static_cast<int>(inter_pts.size());
    orientation(o_0, &inter_pts[0], num_intersections, E_0);
    orientation(o_1, &inter_pts[0], num_intersections, E_1);

    // Sum the segment areas.
    double area = 0;
    for (int i = 0, j = num_intersections - 1; i < num_intersections; j = i++)
    {
      double theta_0 = o_0[j];
      double theta_1 = o_0[i];

      double psi_0 = o_1[j];
      double psi_1 = o_1[i];

      if (theta_0 > theta_1)
        theta_1 += 2 * M_PI;

      if (psi_0 > psi_1)
        psi_1 += 2 * M_PI;

      area += std::min(segment_area(E_0, theta_0, theta_1),
                       segment_area(E_1, psi_0, psi_1));
    }

    // If the number of the intersection > 2, add the area of the polygon
    // whose vertices are p[0], p[1], ..., p[numInter].
    if (num_intersections > 2)
    {
      for (int i = 0, j = num_intersections - 1; i < num_intersections; j = i++)
      {
        Matrix2d M;
        M.col(0) = inter_pts[j];
        M.col(1) = inter_pts[i];
        area += 0.5 * M.determinant();
      }
    }

    return area;
  }

  double analytic_jaccard_similarity(const Ellipse& e1, const Ellipse& e2)
  {
    double interArea = analytic_intersection_area(e1, e2);
    double unionArea = area(e1) + area(e2) - interArea;
    return interArea / unionArea;
  }


  // ======================================================================== //
  // Approximate computation of area of intersection ellipses.
  // ======================================================================== //
  std::vector<Point2d> discretize_ellipse(const Ellipse& e, int n)
  {
    std::vector<Point2d> polygon;
    polygon.reserve(n);

    const Matrix2d Ro(rotation2(e.orientation()));
    Vector2d D(e.radius1(), e.radius2());

    for (int i = 0; i < n; ++i)
    {
      const double theta = 2. * M_PI * double(i) / n;
      const Matrix2d R(rotation2(theta));
      Point2d p(1.0, 0.0);

      const Point2d p1(e.center() +
                       Ro.matrix() * D.asDiagonal() * R.matrix() * p);
      polygon.push_back(p1);
    }

    return polygon;
  }

  std::vector<Point2d> approximate_intersection(const Ellipse& e1,
                                                const Ellipse& e2, int n)
  {
    std::vector<Point2d> p1(discretize_ellipse(e1, n));
    std::vector<Point2d> p2(discretize_ellipse(e2, n));

    std::vector<Point2d> inter;
    inter = sutherland_hodgman(p1, p2);
    if (inter.empty())
      inter = sutherland_hodgman(p2, p1);
    return inter;
  }

  double approximate_jaccard_similarity(const Ellipse& e1, const Ellipse& e2,
                                        int n, double /* limit */)
  {
    std::vector<Point2d> p1(discretize_ellipse(e1, n));
    std::vector<Point2d> p2(discretize_ellipse(e2, n));
    std::vector<Point2d> inter;
    inter = sutherland_hodgman(p1, p2);
    if (inter.empty())
      inter = sutherland_hodgman(p2, p1);

    if (!inter.empty())
    {
      double interArea = area(inter);
      double unionArea = area(p1) + area(p2) - interArea;
      return interArea / unionArea;
    }
    else
      return 0.;
  }

}}  // namespace DO::Sara
