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

#include <drafts/Halide/Utilities.hpp>


namespace DO::Shakti::HalideBackend {

  template <typename FuncOrBuffer>
  auto schedule_histograms(FuncOrBuffer& f,                               //
                           const Halide::Var& o, const Halide::Var& k,    //
                           const Halide::Var& oo, const Halide::Var& ko,  //
                           const Halide::Var& oi, const Halide::Var& ki,  //
                           std::int32_t tile_o, std::int32_t tile_k,      //
                           const Halide::Target& target)                  //
  {
    // GPU schedule.
    if (target.has_gpu_feature())
    {
      f.gpu_tile(o, k, oo, ko, oi, ki, tile_o, tile_k,
                 Halide::TailStrategy::GuardWithIf);
    }

    // Hexagon schedule.
    else if (target.features_any_of(
                 {Halide::Target::HVX_64, Halide::Target::HVX_128}))
    {
      const auto vector_size =
          target.has_feature(Halide::Target::HVX_128) ? 128 : 64;

      f.hexagon()
          .split(k, ko, ki, 128)
          .parallel(ko)
          .vectorize(oi, vector_size, Halide::TailStrategy::GuardWithIf);
    }

    // CPU schedule.
    else
    {
      f.split(k, ko, ki, 4, Halide::TailStrategy::GuardWithIf)
          .parallel(k)
          .vectorize(o, 8);  // No need for bound checks here.
    }
  }

  auto compute_histogram_of_gradients(
      const Halide::Func& mag_fn_ext,                        //
      const Halide::Func& ori_fn_ext,                        //
      const Halide::Expr& x,                                 //
      const Halide::Expr& y,                                 //
      const Halide::Expr& scale,                             //
      const Halide::Expr& scale_max,                         //
      const Halide::Var& o,                                  //
      const Halide::Expr& N = 36,                            //
      const Halide::Expr& gaussian_truncation_factor = 3.f,  //
      const Halide::Expr& scale_mult_factor = 1.5f)          //
      -> Halide::Expr
  {
    const auto sigma_max = scale_mult_factor * scale_max;
    const auto patch_radius_max = Halide::cast<std::int32_t>(
        Halide::floor(sigma_max * gaussian_truncation_factor));

    const auto sigma = scale_mult_factor * scale;
    const auto patch_radius = Halide::cast<std::int32_t>(
        Halide::floor(sigma * gaussian_truncation_factor));

    const auto rounded_x = Halide::cast<std::int32_t>(Halide::round(x));
    const auto rounded_y = Halide::cast<std::int32_t>(Halide::round(y));

    // Define the maximum reduction domain.
    auto r = Halide::RDom(                             //
        -patch_radius_max, 2 * patch_radius_max + 1,   // x
        -patch_radius_max, 2 * patch_radius_max + 1);  // y
    // Define the shape of the reduction domain.
    //
    // The following is more accurate:
    // r.where(r.x * r.x + r.y * r.y < patch_radius * patch_radius);
    //
    // The following is however more efficient and introduces a negligible bias:
    r.where(Halide::abs(r.x) < patch_radius &&  //
            Halide::abs(r.y) < patch_radius);

    // Collect the gradient magnitudes and orientations in the reduction domain.
    const auto mag = mag_fn_ext(rounded_x + r.x, rounded_y + r.y);
    const auto ori = ori_fn_ext(rounded_x + r.x, rounded_y + r.y);

    // Calculate the Gaussian weight of each gradient orientation based on the
    // distance to the patch center.
    const auto ori_weight = Halide::exp(  //
        -(r.x * r.x + r.y * r.y) /        //
        (2 * sigma * sigma));             //

    // Find the corresponding orientation bin for each gradient orientation.
    //
    // Re-express the orientation value in [0, 2π].
    constexpr auto two_pi = static_cast<float>(2 * M_PI);
    const auto ori_in_0_2_pi = Halide::select(ori < 0,       //
                                              ori + two_pi,  //
                                              ori);          //
    // Normalize in [0, 1].
    const auto ori_01_normalized = ori_in_0_2_pi / two_pi;

    // Discretize the interval in [0, 2π[ in 36 smaller intervals centered in:
    // {  0 deg,         10 deg,         20 deg, ...,    350 deg }
    // Each bin cover the following intervals:
    //  [-5, 5[,         [5, 15[,      [15, 25[, ..., [345, 355[
    //
    // Formalising a bit, this becomes:
    // { 0 rad, Δ rad, 2Δ rad, ..., (N - 1)Δ rad }
    //
    // Each interval length is equal to:
    // Δ = 2π / N
    //
    //
    // The intervals are:
    //  [     0  Δ - Δ/2,       0 Δ + Δ/2[,
    //  [     1  Δ - Δ/2,       1 Δ + Δ/2[,
    //  [     2  Δ - Δ/2,       2 Δ + Δ/2[
    //  ...
    //  [(N - 1) Δ - Δ/2, (N - 1) Δ + Δ/2[

#ifdef ORIGINAL_LOWE_IMPL
    // Find the best orientation bin index via simple rounding.
    auto ori_bin = Halide::cast<int>(Halide::round(ori_01_normalized * N));
    // Clamp the orientation bin index via cheap modulo operation.
    ori_bin = Halide::select(ori_bin == N, 0, ori_bin);

    // Accumulation rule.
    const auto is_bin_o = (ori_bin == o);

    return Halide::sum(ori_weight * mag * is_bin_o);
#else
    // It makes more sense to use interpolation.
    const auto ori_0N = ori_01_normalized * N;

    auto ori_0N_floor = Halide::floor(ori_0N);
    ori_0N_floor = Halide::select(ori_0N_floor == N, 0, ori_0N_floor);

    const auto ori_0N_ceil =
        Halide::select(ori_0N_floor == N - 1, 0, ori_0N_floor + 1);

    const auto ori_0N_frac = Halide::fract(ori_0N);

    // Accumulation rule using linear interpolation.
    const auto interpolation_weight = Halide::select(  //
        o == ori_0N_floor,                             //
        (1 - ori_0N_frac),                             //
        Halide::select(o == ori_0N_ceil,               //
                       ori_0N_frac,                    //
                       0));                            //

    return Halide::sum(ori_weight * mag * interpolation_weight);
#endif
  }

  auto compute_histogram_of_gradients(
      const Halide::Func& mag_fn_ext,                        //
      const Halide::Func& ori_fn_ext,                        //
      const Halide::Expr& x,                                 //
      const Halide::Expr& y,                                 //
      const Halide::Expr& t,                                 //
      const Halide::Expr& n,                                 //
      const Halide::Expr& scale,                             //
      const Halide::Expr& scale_max,                         //
      const Halide::Var& o,                                  //
      const Halide::Expr& N = 36,                            //
      const Halide::Expr& gaussian_truncation_factor = 3.f,  //
      const Halide::Expr& scale_mult_factor = 1.5f)          //
      -> Halide::Expr
  {
    const auto sigma_max = scale_mult_factor * scale_max;
    const auto patch_radius_max = Halide::cast<std::int32_t>(
        Halide::floor(sigma_max * gaussian_truncation_factor));

    const auto sigma = scale_mult_factor * scale;
    const auto patch_radius = Halide::cast<std::int32_t>(
        Halide::floor(sigma * gaussian_truncation_factor));

    const auto rounded_x = Halide::cast<std::int32_t>(Halide::round(x));
    const auto rounded_y = Halide::cast<std::int32_t>(Halide::round(y));

    // Define the maximum reduction domain.
    auto r = Halide::RDom(                             //
        -patch_radius_max, 2 * patch_radius_max + 1,   // x
        -patch_radius_max, 2 * patch_radius_max + 1);  // y
    // Define the shape of the reduction domain.
    //
    // The following is more accurate:
    // r.where(r.x * r.x + r.y * r.y < patch_radius * patch_radius);
    //
    // The following is however more efficient and introduces a negligible bias:
    r.where(Halide::abs(r.x) < patch_radius &&  //
            Halide::abs(r.y) < patch_radius);

    // Collect the gradient magnitudes and orientations in the reduction domain.
    const auto mag = mag_fn_ext(rounded_x + r.x, rounded_y + r.y, t, n);
    const auto ori = ori_fn_ext(rounded_x + r.x, rounded_y + r.y, t, n);

    // Calculate the Gaussian weight of each gradient orientation based on the
    // distance to the patch center.
    const auto ori_weight = Halide::exp(  //
        -(r.x * r.x + r.y * r.y) /        //
        (2 * sigma * sigma));             //

    // Find the corresponding orientation bin for each gradient orientation.
    //
    // Re-express the orientation value in [0, 2π].
    constexpr auto two_pi = static_cast<float>(2 * M_PI);
    const auto ori_in_0_2_pi = Halide::select(ori < 0,       //
                                              ori + two_pi,  //
                                              ori);          //
    // Normalize in [0, 1].
    const auto ori_01_normalized = ori_in_0_2_pi / two_pi;

    // Discretize the interval in [0, 2π[ in 36 smaller intervals centered in:
    // {  0 deg,         10 deg,         20 deg, ...,    350 deg }
    // Each bin cover the following intervals:
    //  [-5, 5[,         [5, 15[,      [15, 25[, ..., [345, 355[
    //
    // Formalising a bit, this becomes:
    // { 0 rad, Δ rad, 2Δ rad, ..., (N - 1)Δ rad }
    //
    // Each interval length is equal to:
    // Δ = 2π / N
    //
    //
    // The intervals are:
    //  [     0  Δ - Δ/2,       0 Δ + Δ/2[,
    //  [     1  Δ - Δ/2,       1 Δ + Δ/2[,
    //  [     2  Δ - Δ/2,       2 Δ + Δ/2[
    //  ...
    //  [(N - 1) Δ - Δ/2, (N - 1) Δ + Δ/2[

#ifdef ORIGINAL_LOWE_IMPL
    // Find the best orientation bin index via simple rounding.
    auto ori_bin = Halide::cast<int>(Halide::round(ori_01_normalized * N));
    // Clamp the orientation bin index via cheap modulo operation.
    ori_bin = Halide::select(ori_bin == N, 0, ori_bin);

    // Accumulation rule.
    const auto is_bin_o = (ori_bin == o);

    return Halide::sum(ori_weight * mag * is_bin_o);
#else
    // It makes more sense to use interpolation.
    const auto ori_0N = ori_01_normalized * N;

    auto ori_0N_floor = Halide::floor(ori_0N);
    ori_0N_floor = Halide::select(ori_0N_floor == N, 0, ori_0N_floor);

    const auto ori_0N_ceil =
        Halide::select(ori_0N_floor == N - 1, 0, ori_0N_floor + 1);

    const auto ori_0N_frac = Halide::fract(ori_0N);

    // Accumulation rule using linear interpolation.
    const auto interpolation_weight = Halide::select(  //
        o == ori_0N_floor,                             //
        (1 - ori_0N_frac),                             //
        Halide::select(o == ori_0N_ceil,               //
                       ori_0N_frac,                    //
                       0));                            //

    return Halide::sum(ori_weight * mag * interpolation_weight);
#endif
  }

  auto box_blur(const Halide::Func& hist,  //
                const Halide::Var& o,      //
                const Halide::Var& k,      //
                const Halide::Expr& N)     //
  {
    const auto o_prev = Halide::select((o - 1) < 0, N - 1, o - 1);
    const auto o_next = Halide::select((o + 1) == N, 0, o + 1);

    return (hist(o_prev, k) +   //
            hist(o, k) +        //
            hist(o_next, k)) /  //
           3.f;
  }


  auto is_peak(const Halide::Func& hist,                     //
               const Halide::Var& o,                         //
               const Halide::Var& k,                         //
               const Halide::Expr& N = 36,                   //
               const Halide::Expr& peak_ratio_thres = 0.8f)  //
  {
    auto r = Halide::RDom(0, N);
    auto global_max = Halide::Func{hist.name() + "_global_max"};
    global_max(k) = Halide::maximum(hist(r, k));
    global_max.compute_root();

    const auto o_prev = Halide::select((o - 1) < 0, N - 1, o - 1);
    const auto o_next = Halide::select((o + 1) == N, 0, o + 1);

    const auto is_local_max = hist(o, k) > hist(o_prev, k) &&  //
                              hist(o, k) > hist(o_next, k);

    const auto is_large_enough = hist(o, k) > peak_ratio_thres * global_max(k);

    return is_local_max && is_large_enough;
  }

  auto estimate_peak_residual(const Halide::Func& hist,    //
                              const Halide::Var& o,        //
                              const Halide::Var& k,        //
                              const Halide::Expr& N = 36)  //
  {
    const auto o_prev = Halide::select((o - 1) < 0, N - 1, o - 1);
    const auto o_next = Halide::select((o + 1) == N, 0, o + 1);

    const auto y0 = hist(o_prev, k);
    const auto y1 = hist(o, k);
    const auto y2 = hist(o_next, k);

    // Denoting the orientation histogram function by f, we perform a 2nd-order
    // Taylor approximation:
    //
    //   f(x+h) = f(x) + f'(x) h + f''(x) h^2 / 2
    //
    // Let us approximate f'(x) and f''(x) using central finite difference.
    const auto fprime = (y2 - y0) / 2.f;
    const auto fsecond = y0 - 2.f * y1 + y2;

    // The residual h maximizes the expression f(x + h), i.e., it also zeroes
    // the derivative of this second order polynomial in variable h.
    // h = -f'(x)/f''(x).
    const auto residual = -fprime / fsecond;

    // Each orientation histogram bin are centered in the following points:
    // {0, 10, 20, ..., 350} as detailed inside the implementation of the
    // function `compute_orientation_histogram`.
    //
    // The reason for this is because we use the rounding operation.
    return residual;
  }

  auto compute_peak_residual_map(const Halide::Func& hist,      //
                                 const Halide::Func& peak_map,  //
                                 const Halide::Var& o,          //
                                 const Halide::Var& k,          //
                                 const Halide::Expr& N = 36)    //
  {
    return Halide::select(peak_map(o, k),                         //
                          estimate_peak_residual(hist, o, k, N),  //
                          0);                                     //
  }

}  // namespace DO::Shakti::HalideBackend
