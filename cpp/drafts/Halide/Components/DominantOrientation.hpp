// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <drafts/Halide/MyHalide.hpp>


namespace {

  using namespace Halide;

  template <typename GradientInput, typename PositionInput, typename Output>
  void compute_orientation_histogram(const GradientInput& grad_polar_coords,
                                     const PositionInput& positions,
                                     Output& orientation_histogram, int N,
                                     const Expr& gaussian_truncation_factor = 3,
                                     const Expr& length_in_scale_unit = 1.5)
  {
    Var k{"k"}, o{"o"};

    auto x = positions(k, 0);
    auto y = positions(k, 1);
    auto s = positions(k, 2);

    orientation_histogram(k, o) = 0;
    auto rounded_x = round(x);
    auto rounded_y = round(y);

    auto sigma_gaussian = length_in_scale_unit * s;
    auto patch_radius = sigma_gaussian * gaussian_truncation_factor;

    auto r = RDom(-patch_radius, 2 * patch_radius + 1,   // x
                  -patch_radius, 2 * patch_radius + 1);  // y

    auto grad_polar_coords_ext =
        BoundaryConditions::constant_exterior(grad_polar_coords, {0, 0});
    auto mag = grad_polar_coords_ext(x + r.x, y + r.y, 0);
    auto ori = grad_polar_coords_ext(x + r.x, y + r.y, 1);

    auto ori_shift = select(ori < 0, ori + 2 * M_PI, ori);

    auto weight = exp(-(r.x * r.x + r.y * r.y) /               //
                      (2 * sigma_gaussian * sigma_gaussian));  //

    auto bin_index = cast<int>(floor(ori_shift / (2 * M_PI) * N));
    auto is_bin_o = select(bin_index == o, 0, 1);

    orientation_histogram(k, o) = sum(weight * mag * is_bin_o);
  }

  void lowe_smooth_histogram(HistogramInput& hist_in, HistogramOutput& hist_out,
                             int N)
  {
    Var k{"k"}, o{"o"};
    auto current = Func();
    auto next = Func();

    for (int i = 0; i < 6; ++i)
    {
      if (i == 0)
        current(k, o) = hist_in(k, o);
      else
        current(k, o) = next(k, o);
      next(k, o) = (current(k, select((o - 1) < 0, N - 1, o - 1)) +  //
                    current(k, o) +                                  //
                    current(k, select((o + 1) == N, 0, o + 1)))      //
                   / 3.f;
    }
    hist_out(k, o) = next(k, o);
  }

  void find_peaks(HistogramInput& hist_in, HistogramOutput& hist_out, int N,
                  const Expr& peak_ratio_thres = 0.8)
  {
    Var k{"k"}, o{"o"};

    auto global_max = Func{};
    auto r = RDom(0, N);
    global_max(k) = maximum(hist_in(k, k + r));

    auto is_local_max = Func{};
    is_local_max(o, k) = maximum(
      hist_in(select((o - 1) < 0, N - 1, o - 1), k),
      hist_in(o, k),
      hist_in(select((o + 1) == N, 0, o + 1), k));

    auto is_large_enough = Func{};
    is_large_enough(k, o) = hist_in(k, o) > peak_ratio_thres * global_max(k);

    hist_out(k, o) = is_large_enough(k, o) && is_local_max(k, o);
  }

  void refine_peak(const HistogramInput& orientation_histogram,
                   HistogramOutput& hist_out, int N)
  {
    Var k{"k"}, o{"o"};

    auto y0 = orientation_histogram((N + o - 1) % N, k);
    auto y1 = orientation_histogram(o, k);
    auto y2 = orientation_histogram((o + 1) % N, k);

    // Denote the orientation histogram function by \f$f\f$.
    // perform a 2nd-order Taylor approximation:
    // \f$f(x+h) = f(x) + f'(x)h + f''(x) h^2/2\f$
    // We approximate \f$f'\f$ and \f$f''\f$ by finite difference.
    auto fprime = (y2-y0) / 2.f;
    auto fsecond = y0 - 2.f*y1 + y2;

    // Maximize w.r.t. to \f$h\f$, derive the expression.
    // Thus \f$h = -f'(x)/f''(x)\f$.
    auto h = -fprime / fsecond;

    // Add the offset \f$h\f$ to get the refined orientation value.
    // Note that we also add the 0.5f offset, because samples are assumed taken
    // on the middle of the interval \f$[i, i+1)\f$.
    hist_out(o, k) = o + 0.5 + h;
  }

}  // namespace
