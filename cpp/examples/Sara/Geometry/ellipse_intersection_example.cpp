// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Visualization.hpp>

#include <random>


using namespace std;
using namespace DO::Sara;

namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  auto w = 400;
  auto h = 400;

  create_window(w, h);
  set_antialiasing();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> coord_dist(0.0, static_cast<double>(w));
  std::uniform_real_distribution<> radius_dist(1.0, static_cast<double>(h));
  std::uniform_real_distribution<> orientation_dist(0., 2 * M_PI);

  for (auto i = 0; i < 1000; ++i)
  {
    clear_window();


    const auto e1 =
        sara::Ellipse{radius_dist(gen), radius_dist(gen), orientation_dist(gen),
                      Point2d(coord_dist(gen), coord_dist(gen))};
    const auto e2 =
        sara::Ellipse{radius_dist(gen), radius_dist(gen), orientation_dist(gen),
                      Point2d(coord_dist(gen), coord_dist(gen))};
    draw_ellipse(e1, Red8, 3);
    draw_ellipse(e2, Blue8, 3);

    const auto intersection_points = compute_intersection_points(e1, e2);

    for (const auto& p : intersection_points)
      fill_circle(p.cast<float>(), 3, Green8);

    // Quad Q_0(oriented_bbox(E_0));
    // Quad Q_1(oriented_bbox(E_1));

    // BBox b0(&Q_0[0], &Q_0[0] + 4);
    // BBox b1(&Q_1[0], &Q_1[0] + 4);
    // b0.top_left() = b0.top_left().cwiseMin(b1.top_left());
    // b0.bottom_right() = b0.bottom_right().cwiseMax(b1.bottom_right());

    // draw_quad(Q_0, Red8, 3);
    // draw_quad(Q_1, Blue8, 3);
    // draw_bbox(b0, Green8, 3);
    // get_key();

    //// now rescale the ellipse.
    // Point2d center(b0.center());
    // Vector2d delta(b0.sizes() - center);
    // delta = delta.cwiseAbs();
    //
    // Ellipse EE_0, EE_1;
    // Matrix2d S_0 = delta.asDiagonal() * shape_matrix(E_0) *
    // delta.asDiagonal(); Matrix2d S_1 = delta.asDiagonal() * shape_matrix(E_1)
    // * delta.asDiagonal();
    //
    // SARA_CHECK(shape_matrix(E_0));
    // SARA_CHECK(S_0);
    //
    // Vector2d c_0 = E_0.center() - center;
    // Vector2d c_1 = E_1.center() - center;
    // EE_0 = construct_from_shape_matrix(S_0, c_0);
    // EE_1 = construct_from_shape_matrix(S_1, c_1);
    // int num_inter = compute_intersection_points(inter_pts, EE_0, EE_1);
    //
    // for (int i = 0; i < num_inter; ++i)
    //  inter_pts[i] = delta.asDiagonal() * inter_pts[i] + center;
    //}
    get_key();
  }

  return 0;
}