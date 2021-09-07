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
#include <DO/Sara/Core/Math/PolynomialRoots.hpp>

#include <DO/Sara/Geometry/Algorithms/ConvexHull.hpp>
#include <DO/Sara/Geometry/Algorithms/EllipseIntersection.hpp>
#include <DO/Sara/Geometry/Algorithms/SutherlandHodgman.hpp>
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
      -> Univariate::UnivariatePolynomial<double, 4>
  {
    double d[6][6];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        d[i][j] = s[i] * t[j] - s[j] * t[i];

    auto u = Univariate::UnivariatePolynomial<double, 4>{};
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
      -> Univariate::UnivariatePolynomial<double, 2>
  {
    auto sigma = Univariate::UnivariatePolynomial<double, 2>{};
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
                              std::vector<Point2d>& intersections) -> void
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

  auto compute_intersection_points(const Ellipse& e1, const Ellipse& e2,
                                   bool polish_intersection_points)
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
    auto ys_as_complex = std::array<std::complex<double>, 4>{};
    roots(u, ys_as_complex[0], ys_as_complex[1], ys_as_complex[2],
          ys_as_complex[3]);

    // Keep the real roots.
    auto ys = std::vector<double>{};
    for (const auto& y_complex : ys_as_complex)
    {
      // We should be close to the real solution if we are dealing with an
      // actual intersection point.
      using std::imag;
      using std::real;

      const auto is_y_real =
          std::abs(imag(y_complex)) < 1e-2 * std::abs(real(y_complex));
      if (!is_y_real)
        continue;

      ys.push_back(std::real(y_complex));
    }

    // Remove redundant roots.
    constexpr auto eps = 1e-4;
    constexpr auto squared_eps = eps * eps;
    std::sort(ys.begin(), ys.end());
    const auto y_last = std::unique(ys.begin(), ys.end(),
                                    [&eps](const double& a, const auto& b) {
                                      return std::abs(a - b) < eps;
                                    });
    ys.resize(y_last - ys.begin());

    // Now polish the roots with Newton-Raphson optimization method if needed.
    if (polish_intersection_points)
    {
      for (auto& y : ys)
      {
        auto P = Univariate::UnivariatePolynomial<double, -1>{};
        for (auto d = 0; d <= u.degree(); ++d)
          P[d] = u[d];

        auto root_polisher = Univariate::NewtonRaphson{P};
        y = root_polisher(y, 10);
      }
    }

    // Finally form the intersection points for which the y-coordinate is real.
    auto intersections = std::vector<Point2d>{};
    for (const auto& y : ys)
      add_intersection_point(s, t, y, intersections);

    // Remove any intersection duplicates.
    const auto intersection_last =
        std::unique(intersections.begin(), intersections.end(),
                    [](const Point2d& a, const Point2d& b) {
                      return (a - b).squaredNorm() < squared_eps;
                    });

    const auto num_intersections = intersection_last - intersections.begin();
    if (num_intersections > 4)
      throw std::runtime_error{
          "Ellipse intersections: too many intersections!"};
    intersections.resize(num_intersections);

    // Go back to the original frame by shifting the intersection points.
    for (auto& p : intersections)
      p += center;

    return intersections;
  }


  // ======================================================================== //
  // Computation of the area of intersecting ellipses.
  // ======================================================================== //
  auto orientation(double* ori, const Point2d* pts, int num_points,
                   const Ellipse& e) -> void
  {
    const auto u = unit_vector2(e.orientation());
    const auto v = Vector2d{-u(1), u(0)};
    std::transform(pts, pts + num_points, ori, [&](const Point2d& p) -> double {
      const Vector2d d = p - e.center();
      return atan2(v.dot(d), u.dot(d));
    });
  }

  double analytic_intersection_area(const Ellipse& e1, const Ellipse& e2,
                                    bool polish_intersection_points)
  {
    // The ellipse intersection is not entirely robust numerically. I will
    // investigate later.
    auto inter_pts =
        compute_intersection_points(e1, e2, polish_intersection_points);

    // TODO: Check these conditions again. We might have to check some corner
    // cases.
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
      if (e1.contains(e2.center()) || e2.contains(e1.center()))
        return std::min(area(e1), area(e2));
    }

    // GENERIC CASE
    //
    // Compute the relative orientation of the intersection points w.r.t. the
    // ellipse orientation.
    double orientations1[4];
    double orientations2[4];
    const auto num_intersections = static_cast<int>(inter_pts.size());
    orientation(orientations1, &inter_pts[0], num_intersections, e1);
    orientation(orientations2, &inter_pts[0], num_intersections, e2);

    // Sum the segment areas.
    double area = 0;
    for (int i = 0, j = num_intersections - 1; i < num_intersections; j = i++)
    {
      double theta_0 = orientations1[j];
      double theta_1 = orientations1[i];

      double psi_0 = orientations2[j];
      double psi_1 = orientations2[i];

      // TODO: Check these conditions again. We might have to check some corner
      // cases.
      if (theta_0 > theta_1)
        theta_1 += 2 * M_PI;

      if (psi_0 > psi_1)
        psi_1 += 2 * M_PI;

      area += std::min(segment_area(e1, theta_0, theta_1),
                       segment_area(e2, psi_0, psi_1));
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

  double analytic_jaccard_similarity(const Ellipse& e1, const Ellipse& e2,
                                     bool polish_intersection_points)
  {
    const auto interArea =
        analytic_intersection_area(e1, e2, polish_intersection_points);
    const auto unionArea = area(e1) + area(e2) - interArea;
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
