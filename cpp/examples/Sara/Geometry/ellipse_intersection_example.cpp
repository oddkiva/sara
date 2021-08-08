// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
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
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> coord_dist(100., 300.);
  std::uniform_real_distribution<> radius_dist(1.0, 150.);
  std::uniform_real_distribution<> orientation_dist(-2 * M_PI, 2 * M_PI);

  auto bad_computations = 0;
  auto total = 10000;
  for (auto i = 0; i < total; ++i)
  {
    const auto e1 =
        sara::Ellipse{radius_dist(gen), radius_dist(gen), orientation_dist(gen),
                      Point2d(coord_dist(gen), coord_dist(gen))};
    const auto e2 =
        sara::Ellipse{radius_dist(gen), radius_dist(gen), orientation_dist(gen),
                      Point2d(coord_dist(gen), coord_dist(gen))};

    const auto intersection_points = compute_intersection_points(e1, e2);

    const auto inter_area_analytic = sara::analytic_intersection_area(e1, e2);
    const auto inter_area_approx =
        sara::area(sara::approximate_intersection(e1, e2, 360));

    const auto diff = std::abs(inter_area_analytic - inter_area_approx);
    const auto diff_relative =
        inter_area_approx < 1e-2 ? 0 : diff / inter_area_approx;
    const auto good = diff_relative < 1e-2;

    SARA_CHECK(i);
    SARA_CHECK(diff_relative);
    SARA_CHECK(inter_area_analytic);
    SARA_CHECK(inter_area_approx);
    SARA_DEBUG << (good ? "OK" : "KOOOOOOOOOOOOOO") << std::endl;

//#define INSPECT_VISUALLY
#ifdef INSPECT_VISUALLY
    if (!active_window())
    {
      create_window(400, 400);
      set_antialiasing();
    }

    clear_window();
    draw_ellipse(e1, Red8, 3);
    draw_ellipse(e2, Blue8, 3);
    for (const auto& p : intersection_points)
      fill_circle(p.cast<float>(), 3.f, Green8);
    if (!good)
      get_key();
#endif

    bad_computations += int(!good);
  }

  // One run of the program shows that 24 random instances out of 10,000 will
  // show that the relative difference between the polygonal approach and the
  // analytic approach is more than 1%.
  //
  // There are still some rare corner cases where the analytical form is very wrong.
  // Find out where in the formula.
  SARA_CHECK(bad_computations);
  SARA_CHECK(bad_computations / double(total));

  return 0;
}