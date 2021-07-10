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

#include <DO/Shakti/Halide/MyHalide.hpp>
#include <DO/Shakti/Halide/Components/TinyLinearAlgebra.hpp>


namespace DO::Shakti::HalideBackend {

  struct SIFT
  {
    static constexpr auto two_pi = static_cast<float>(2 * M_PI);

    //! @brief SIFT build histograms from a grid of NxN patches of image
    //! gradients.
    static constexpr std::int32_t N = 4;
    //! @brief Each histogram of gradients consists of O = 8 orientation bins.
    /*!
     *  Each bin are centered around the orientations:
     *  bin index   = [0,   1,    2,    3, 4,    5,    6,    7]
     *  orientation = [0, π/4,  π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
     */
    static constexpr std::int32_t O = 8;

    //! @brief Each image subpatch has a radius equal to this value:
    static constexpr auto bin_length_in_scale_unit = 3.f;
    //! @brief Each image subpatch has a diameter equal to this value:
    static constexpr auto subpatch_diameter = 2 * bin_length_in_scale_unit + 1;

    //! @brief The magic constant value to enforce invariance to nonlinear
    //! illumination changes as referenced in the paper.
    static constexpr auto max_bin_value = 0.2f;

    Halide::Func contrast_norm{"contrast_change_norm"};
    Halide::Func illumination_norm{"illumination_norm"};
    Halide::Func hist_contrast_invariant{"hist_contrast_invariant"};
    Halide::Func hist_clamped{"hist_clamped"};
    Halide::Func hist_illumination_invariant{"hist_illumination_invariant"};

    //! @brief Radius of the whole image patch.
    auto patch_radius(const Halide::Expr& scale) const
    {
      return bin_length_in_scale_unit * scale *  //
             (N + 1) / 2.f * std::sqrt(2.f);
    }

    //! @brief Reduction domain associated to the whole image patch.
    auto reduction_domain(const Halide::Expr& scale,
                          const Halide::Expr& scale_max) const
    {
      // Calculate the radius upper-bound.
      const auto r_max =
          Halide::cast<std::int32_t>(Halide::round(patch_radius(scale_max)));
      // Calculate the radius of the actual patch.
      const auto r_actual =
          Halide::cast<std::int32_t>(Halide::round(patch_radius(scale)));

      // Define the reduction domain.
      auto r = Halide::RDom(-r_max, 2 * r_max + 1, -r_max, 2 * r_max + 1);
      // The actual shape of the reduction domain is:
      r.where(Halide::abs(r.x) < r_actual &&  //
              Halide::abs(r.y) < r_actual);

      return r;
    }

    //! @brief Radius of each image sub-patch (i, j).
    auto subpatch_radius(const Halide::Expr& scale) const
    {
      return bin_length_in_scale_unit * scale * std::sqrt(2.f);
    }

    //! @brief Radius for each image sub-patch (i, j).
    auto subpatch_reduction_domain(const Halide::Expr& scale,
                                   const Halide::Expr& scale_max) const
    {
      // Calculate the radius upper-bound.
      const auto r_max =
          Halide::cast<std::int32_t>(Halide::round(subpatch_radius(scale_max)));
      // Calculate the radius of the actual patch.
      const auto r_actual =
          Halide::cast<std::int32_t>(Halide::round(subpatch_radius(scale)));

      // Define the reduction domain.
      auto r = Halide::RDom(-r_max, 2 * r_max + 1, -r_max, 2 * r_max + 1);
      // The actual shape of the reduction domain is:
      r.where(Halide::abs(r.x) < r_actual &&  //
              Halide::abs(r.y) < r_actual);

      return r;
    }

    //! @brief Weight associated to each sampled image gradient of the whole
    //! image patch.
    auto gradient_weight(const Halide::Var& u, const Halide::Var& v) const
    {
      // Calculate the standard deviation of the gaussian weight:
      constexpr auto sigma = N / 2 * bin_length_in_scale_unit;
      // which evaluates to: 4 / 2 * 3 = 2

      return Halide::exp(-(u * u + v * v) / (2 * Halide::pow(sigma, 2)));
    }

    //! @brief Gradient sample at the normalized image patch.
    auto normalized_gradient_sample(const Halide::Var& u,                //
                                    const Halide::Var& v,                //
                                    const Halide::Func& grad_mag_fn,     //
                                    const Halide::Func& grad_ori_fn,     //
                                    const Halide::Func& grad_weight_fn,  //
                                    const Halide::Expr& x,               //
                                    const Halide::Expr& y,               //
                                    const Halide::Expr& scale,           //
                                    const Halide::Expr& theta) const
    {
      // Linear part of the patch normalization transform.
      const auto c = Halide::cos(theta);
      const auto s = Halide::sin(theta);

      auto T = Matrix2{};
      T(0, 0) = c; T(0, 1) = -s;
      T(1, 0) = s; T(1, 1) = c;
      T *= scale;

      // For each point of the normalized patch, i.e.:
      auto p = Vector2{};
      p(0) = Halide::cast<float>(u);
      p(1) = Halide::cast<float>(v);

      // Find the corresponding point in the image:
      const auto Tp = T * p;
      const auto Tu = Halide::cast<std::int32_t>(Halide::round(x + Tp(0)));
      const auto Tv = Halide::cast<std::int32_t>(Halide::round(y + Tp(1)));

      // Gradient orientation w.r.t. the dominant gradient orientation.
      auto ori_normalized = grad_ori_fn(Tu, Tv) - theta;
      // Re-express the orientation in [0, 2π[.
      ori_normalized = Halide::select(ori_normalized < 0,       //
                                      ori_normalized + two_pi,  //
                                      ori_normalized);
      // Re-express the orientation in [0, O[.
      ori_normalized *= O / two_pi;

      // The canonical patch of normalized image gradients.
      return Halide::Tuple{
          grad_weight_fn(u, v) * grad_mag_fn(Tu, Tv),  //
          ori_normalized,                            //
      };
    }

    //! Precompute the spatial weights used in the trilinear interpolation.
    auto spatial_weight(const Halide::Var& x, //
                        const Halide::Var& y) const
    {
      // Spatial weights.
      const auto dx = Halide::cast<float>(Halide::abs(x)) / bin_length_in_scale_unit;
      const auto dy = Halide::cast<float>(Halide::abs(y)) / bin_length_in_scale_unit;
      // Thus:
      const auto wx = Halide::max(1.f - dx, 0.f);
      const auto wy = Halide::max(1.f - dy, 0.f);

      return wx * wy;
    }

    //! @brief Calculate the value of the histogram bin (i, j, o).
    //! Contains numerical errors.
    //!
    //! 100000 times slower than the CPU implementation...
    auto compute_bin_value_v1(const Halide::Var& i,             //
                              const Halide::Var& j,             //
                              const Halide::Var& o,             //
                              const Halide::Func& grad_mag_fn,  //
                              const Halide::Func& grad_ori_fn,  //
                              const Halide::Expr& x,            //
                              const Halide::Expr& y,            //
                              const Halide::Expr& scale,        //
                              const Halide::Expr& scale_max,    //
                              const Halide::Expr& theta) const  //
    {
      // Define the reduction domain.
      auto r = reduction_domain(scale, scale_max);

      // The gradient magnitude is:
      const auto xi = Halide::cast<std::int32_t>(Halide::round(x));
      const auto yi = Halide::cast<std::int32_t>(Halide::round(y));
      const auto mag = grad_mag_fn(xi + r.x, yi + r.y);

      // The orientation in the reoriented patch is:
      const auto ori_shifted = grad_ori_fn(xi + r.x, yi + r.y) - theta;
      // We have:
      // -π < ori < π
      // -π < -theta < π
      // -2π < ori - theta < 2π
      // So we make sure that the orientation is [0, 2π[ as follows:
      const auto ori = Halide::select(ori_shifted < 0,       //
                                      ori_shifted + two_pi,  //
                                      ori_shifted);          //
      const auto ori_normalized = ori / two_pi;
      const auto ori_index = ori_normalized * O;

      // Linear part of the patch normalization transform.
      const Halide::Expr bin_length_in_pixels = bin_length_in_scale_unit * scale;
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= bin_length_in_pixels;

      // For each point of the patch, i.e.:
      auto p = Vector2{};
      p(0) = Halide::cast<float>(r.x);
      p(1) = Halide::cast<float>(r.y);

      // Calculate the coordinates of the gradient in the reoriented normalized
      // patch.
      const auto Tp = T * p;  // 1. Apply the patch normalization transform.

      // The weight of this gradient is:
      const auto weight = Halide::exp(-squared_norm(Tp) /  //
                                      (2 * Halide::pow(N / 2.f, 2)));

      auto Tp2 = Vector2{};   // 2. Find out which bin (i, j) it belongs to.
      Tp2(0) = Tp(0) + N / 2.f - 0.5f;
      Tp2(1) = Tp(1) + N / 2.f - 0.5f;

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j, o).
      auto dx = Halide::abs(Halide::cast<float>(j) - Tp2(0));
      auto dy = Halide::abs(Halide::cast<float>(i) - Tp2(1));
      auto dori = Halide::abs(Halide::cast<float>(o) - ori_index);
      // Accumulation rule based on trilinear interpolation.
      auto wx = Halide::select(dx < 1, 1 - dx, 0);
      auto wy = Halide::select(dy < 1, 1 - dy, 0);
      auto wo = Halide::select(dori < 1, 1 - dori, 0);

      return Halide::sum(wx * wy * wo * mag);
    }

    //! @brief Calculate the value of the histogram bin (i, j, o).
    //! Contains numerical errors.
    //!
    //! 100000 times slower than the CPU implementation...
    auto compute_bin_value_v2(const Halide::Var& i,             //
                              const Halide::Var& j,             //
                              const Halide::Var& o,             //
                              const Halide::Func& grad_mag_fn,  //
                              const Halide::Func& grad_ori_fn,  //
                              const Halide::Expr& x,            //
                              const Halide::Expr& y,            //
                              const Halide::Expr& scale,        //
                              const Halide::Expr& scale_max,    //
                              const Halide::Expr& theta) const  //
    {
      // Define the reduction domain of the subpatch (i, j).
      auto r = subpatch_reduction_domain(scale, scale_max);

      // Calculate the coordinates of the patch center (i, j).
      // i                     =   0        1        2        3
      // i_centered            =  -2       -1        0        1
      // intervals             = [-2.0, -1.0] [-1.0, +0.0] [+0.0, +1.0] [+1.0, +2.0]
      // center                =    -1.5         -0.5         +0.5         +1.5
      // overlapping intervals = [-2.5, -0.5] [-1.5, +0.5] [-0.5, +1.5] [+0.5, +2.5]
      const Halide::Expr bin_length_in_pixels =
          bin_length_in_scale_unit * scale;
      const auto dx_ij = cos(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(j) - N / 2.f + 0.5f);
      const auto dy_ij = sin(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(i) - N / 2.f + 0.5f);

      // The gradient magnitude is:
      const auto x_ij = Halide::cast<std::int32_t>(Halide::round(x + dx_ij));
      const auto y_ij = Halide::cast<std::int32_t>(Halide::round(y + dy_ij));
      const auto mag = grad_mag_fn(x_ij + r.x, y_ij + r.y);

      // The orientation in the reoriented patch is:
      const auto ori_shifted = grad_ori_fn(x_ij + r.x, y_ij + r.y) - theta;
      // We have:
      // -π < ori < π
      // -π < -theta < π
      // -2π < ori - theta < 2π
      // So we make sure that the orientation is [0, 2π[ as follows:
      const auto ori = Halide::select(ori_shifted < 0,       //
                                      ori_shifted + two_pi,  //
                                      ori_shifted);          //
      const auto ori_normalized = ori / two_pi;
      const auto ori_index = ori_normalized * O;

      // For each point of the patch, i.e.:
      auto p = Vector2{};
      p(0) = x_ij + Halide::cast<float>(r.x) - x;
      p(1) = y_ij + Halide::cast<float>(r.y) - y;

      // Define the patch normalization transform.
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= bin_length_in_pixels;

      // Calculate the coordinates of the gradient in the reoriented normalized
      // patch.
      const auto Tp = T * p;  // 1. Apply the patch normalization transform.

      // The weight of this gradient is:
      const auto weight = Halide::exp(-squared_norm(Tp) /  //
                                      (2 * Halide::pow(N / 2.f, 2)));

      auto Tp2 = Vector2{};   // 2. Find out which bin (i, j) it belongs to.
      Tp2(0) = Tp(0) + N / 2.f - 0.5f;
      Tp2(1) = Tp(1) + N / 2.f - 0.5f;

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j, o).
      auto dx = Halide::abs(Halide::cast<float>(j) - Tp2(0));
      auto dy = Halide::abs(Halide::cast<float>(i) - Tp2(1));
      auto dori = Halide::abs(Halide::cast<float>(o) - ori_index);
      // Accumulation rule based on trilinear interpolation.
      auto wx = Halide::select(dx < 1, 1 - dx, 0);
      auto wy = Halide::select(dy < 1, 1 - dy, 0);
      auto wo = Halide::select(dori < 1, 1 - dori, 0);

      return Halide::sum(wx * wy * wo * mag);
    }

    //! @brief Calculate the value of the histogram bin (i, j, o).
    //!
    //! The right approach for GPUs in terms of speed, but contains numerical
    //! errors.
    auto compute_bin_value_v3(const Halide::Expr& o,                      //
                              const Halide::Expr& j,                      //
                              const Halide::Expr& i,                      //
                              const Halide::Expr& k,                      //
                              const Halide::Func& grad_normalized) const  //
    {
      auto r = Halide::RDom(                                            //
          -bin_length_in_scale_unit, 2 * bin_length_in_scale_unit + 1,  //
          -bin_length_in_scale_unit, 2 * bin_length_in_scale_unit + 1   //
      );

      const auto x =
          Halide::round((j - N / 2 + 0.5f) * bin_length_in_scale_unit);
      const auto y =
          Halide::round((i - N / 2 + 0.5f) * bin_length_in_scale_unit);

      const auto xi = Halide::cast<std::int32_t>(x);
      const auto yi = Halide::cast<std::int32_t>(y);

      const auto mag = grad_normalized(xi + r.x, yi + r.y, k)[0];
      const auto ori = grad_normalized(xi + r.x, yi + r.y, k)[1];

      const auto dori = Halide::abs(ori - o);
      // OUCH: becomes incorrect when: when o == 0 or o == 7.

      const auto wx = 1.f - Halide::abs(Halide::cast<float>(r.x)) / bin_length_in_scale_unit;
      const auto wy = 1.f - Halide::abs(Halide::cast<float>(r.y)) / bin_length_in_scale_unit;
      const auto wo = Halide::select(dori < 1, 1 - dori, 0);

      return Halide::sum(wo * wy * wx * mag);
    }

    //! @brief Calculate the whole unnormalized SIFT descriptor.
    //! Numerically correct.
    //!
    //! 10000 times slower than the CPU implementation...
    template <typename FuncOrBuffer>
    auto accumulate_histogram(FuncOrBuffer& h,                  //
                              Halide::Var& k,                   //
                              const Halide::Func& grad_mag_fn,  //
                              const Halide::Func& grad_ori_fn,  //
                              const Halide::Expr& x,            //
                              const Halide::Expr& y,            //
                              const Halide::Expr& scale,        //
                              const Halide::Expr& scale_max,    //
                              const Halide::Expr& theta) const  //
    {
      auto r = reduction_domain(scale, scale_max);

      // The gradient magnitude is:
      const auto xi = Halide::cast<std::int32_t>(Halide::round(x));
      const auto yi = Halide::cast<std::int32_t>(Halide::round(y));
      const auto mag = grad_mag_fn(xi + r.x, yi + r.y);

      // The orientation in the reoriented patch is:
      const auto ori_shifted = grad_ori_fn(xi + r.x, yi + r.y) - theta;
      // We have:
      // -π < ori < π
      // -π < -theta < π
      // -2π < ori - theta < 2π
      // So we make sure that the orientation is [0, 2π[ as follows:
      const auto ori = Halide::select(ori_shifted < 0,       //
                                      ori_shifted + two_pi,  //
                                      ori_shifted);          //
      const auto ori_normalized = ori / two_pi;
      const auto ori_index = ori_normalized * O;

      // Linear part of the patch normalization transform.
      const Halide::Expr bin_length_in_pixels = bin_length_in_scale_unit * scale;
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= bin_length_in_pixels;

      // For each point of the patch, i.e.:
      auto p = Vector2{};
      p(0) = Halide::cast<float>(r.x);
      p(1) = Halide::cast<float>(r.y);

      // Calculate the coordinates of the gradient in the reoriented normalized
      // patch.
      const auto Tp = T * p;  // 1. Apply the patch normalization transform.

      // The weight of this gradient is:
      const auto weight = Halide::exp(-squared_norm(Tp) /  //
                                      (2 * Halide::pow(N / 2.f, 2)));

      auto Tp2 = Vector2{};   // 2. Find out which bin (i, j) it belongs to.
      Tp2(0) = Tp(0) + N / 2.f - 0.5f;
      Tp2(1) = Tp(1) + N / 2.f - 0.5f;

      const auto i_int = Halide::cast<int>(Tp2(1));
      const auto j_int = Halide::cast<int>(Tp2(0));
      const auto o_int = Halide::cast<int>(ori_index);

      const auto i0 = Halide::clamp(i_int, 0, N -1);
      const auto j0 = Halide::clamp(j_int, 0, N -1);
      const auto& o0 = o_int;

      const auto i1 = Halide::clamp(i_int + 1, 0, N - 1);
      const auto j1 = Halide::clamp(j_int + 1, 0, N - 1);
      const auto o1 = select(o0 == O - 1, 0, o0 + 1);

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j, o).
      const auto dy = Halide::fract(Tp2(1));
      const auto dx = Halide::fract(Tp2(0));
      const auto dori = Halide::fract(ori_index);

      // Accumulation rule based on trilinear interpolation.
      auto wx0 = Halide::select(j_int < 0, 0, 1 - dx);
      auto wx1 = Halide::select(j_int + 1 >= N, 0, dx);

      auto wy0 = Halide::select(i_int < 0, 0, 1 - dy);
      auto wy1 = Halide::select(i_int + 1 >= N, 0, dy);

      auto wo0 = 1 - dori;
      auto wo1 = dori;

      auto at = [](const auto& i, const auto& j, const auto& o) {
        return Halide::clamp(i * N * O + j * O + o, 0, N * N * O - 1);
      };

      // 8 accumulation schemes.
      h(at(i0, j0, o0), k) += wy0 * wx0 * wo0 * weight * mag;
      h(at(i0, j0, o1), k) += wy0 * wx0 * wo1 * weight * mag;

      h(at(i0, j1, o0), k) += wy0 * wx1 * wo0 * weight * mag;
      h(at(i0, j1, o1), k) += wy0 * wx1 * wo1 * weight * mag;

      h(at(i1, j0, o0), k) += wy1 * wx0 * wo0 * weight * mag;
      h(at(i1, j0, o1), k) += wy1 * wx0 * wo1 * weight * mag;

      h(at(i1, j1, o0), k) += wy1 * wx1 * wo0 * weight * mag;
      h(at(i1, j1, o1), k) += wy1 * wx1 * wo1 * weight * mag;
    }

    //! @brief Calculate the histogram of gradients for each subpatch (i, j).
    //! Numerically correct.
    //!
    //! 10000 times slower than the CPU implementation...
    template <typename FuncOrBuffer>
    auto accumulate_subhistogram(FuncOrBuffer& h,                  //
                                 Halide::Var& ji,                  //
                                 Halide::Var& k,                   //
                                 const Halide::Func& grad_mag_fn,  //
                                 const Halide::Func& grad_ori_fn,  //
                                 const Halide::Expr& x,            //
                                 const Halide::Expr& y,            //
                                 const Halide::Expr& scale,        //
                                 const Halide::Expr& scale_max,    //
                                 const Halide::Expr& theta) const  //
    {
      // Define the reduction domain of the subpatch (i, j).
      auto r = subpatch_reduction_domain(scale, scale_max);

      // Retrieve the (i, j) coordinates of the corresponding image subpatch.
      const auto i = ji / N;
      const auto j = ji % N;

      // Calculate the coordinates of the patch center (i, j).
      //   0        1        2        3
      //  -2       -1        0        1
      // [-2, -1] [-1, -0] [+0, +1] [+1, +2]
      //  -1.5     -0.5     +0.5     +1.5
      const Halide::Expr bin_length_in_pixels =
          bin_length_in_scale_unit * scale;
      const auto dx_ij = cos(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(j) - N / 2.f + 0.5f);
      const auto dy_ij = sin(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(i) - N / 2.f + 0.5f);

      // The gradient magnitude is:
      const auto x_ij = Halide::cast<std::int32_t>(Halide::round(x + dx_ij));
      const auto y_ij = Halide::cast<std::int32_t>(Halide::round(y + dy_ij));
      const auto mag = grad_mag_fn(x_ij + r.x, y_ij + r.y);

      // The orientation in the reoriented patch is:
      const auto ori_shifted = grad_ori_fn(x_ij + r.x, y_ij + r.y) - theta;
      // We have:
      // -π < ori < π
      // -π < -theta < π
      // -2π < ori - theta < 2π
      // So we make sure that the orientation is [0, 2π[ as follows:
      const auto ori = Halide::select(ori_shifted < 0,       //
                                      ori_shifted + two_pi,  //
                                      ori_shifted);          //
      const auto ori_normalized = ori / two_pi;
      const auto ori_index = ori_normalized * O;

      // For each point of the patch, i.e.:
      auto p = Vector2{};
      p(0) = x_ij + Halide::cast<float>(r.x) - x;
      p(1) = y_ij + Halide::cast<float>(r.y) - y;

      // Define the patch normalization transform.
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= bin_length_in_pixels;

      // Calculate the coordinates of the gradient in the reoriented normalized
      // patch.
      const auto Tp = T * p;  // 1. Apply the patch normalization transform.

      // The weight of this gradient is:
      const auto weight = Halide::exp(-squared_norm(Tp) /  //
                                      (2 * Halide::pow(N / 2.f, 2)));

      auto Tp2 = Vector2{};   // 2. Find out which bin (i, j) it belongs to.
      Tp2(0) = Tp(0) + N / 2.f - 0.5f;
      Tp2(1) = Tp(1) + N / 2.f - 0.5f;

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j).
      const auto dx = Halide::abs(Halide::cast<float>(j) - Tp2(0));
      const auto dy = Halide::abs(Halide::cast<float>(i) - Tp2(1));
      // Accumulation rule based on trilinear interpolation.
      const auto wx = Halide::select(dx < 1, 1 - dx, 0);
      const auto wy = Halide::select(dy < 1, 1 - dy, 0);

      const auto o_int = Halide::cast<int>(ori_index);
      const auto o0 = Halide::clamp(                     //
          Halide::select(o_int >= O, o_int - O, o_int),  //
          0,                                             //
          O - 1);
      const auto o1 = Halide::clamp(               //
          Halide::select(o0 == O - 1, 0, o0 + 1),  //
          0,                                       //
          O - 1);

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j, o).
      const auto dori = Halide::fract(ori_index);

      auto wo0 = 1 - dori;
      auto wo1 = dori;

      h(o0, ji, k) += wo0 * wx * wy * weight * mag;
      h(o1, ji, k) += wo1 * wx * wy * weight * mag;
    }

    //! @brief Calculate the histogram of gradients for each subpatch (i, j).
    //! Numerically correct.
    //!
    //! 10000 times slower than the CPU implementation...
    template <typename FuncOrBuffer>
    auto accumulate_subhistogram_v2(FuncOrBuffer& h,                     //
                                    Halide::Var& ji,                     //
                                    Halide::Var& k,                      //
                                    const Halide::Func& gradient_patch,  //
                                    const Halide::Expr& x,               //
                                    const Halide::Expr& y,               //
                                    const Halide::Expr& scale,           //
                                    const Halide::Expr& scale_max,       //
                                    const Halide::Expr& theta) const     //
    {
      // Define the reduction domain of the subpatch (i, j).
      const auto r = subpatch_reduction_domain(scale, scale_max);

      auto mag_fn = gradient_patch(r.x, r.y, ji, k)[0];
      auto ori_fn = gradient_patch(r.x, r.y, ji, k)[1];

      // Retrieve the (i, j) coordinates of the corresponding image subpatch.
      const auto i = ji / N;
      const auto j = ji % N;

      const Halide::Expr bin_length_in_pixels =
          bin_length_in_scale_unit * scale;
      const auto dx_ij = cos(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(j) - N / 2.f + 0.5f);
      const auto dy_ij = sin(theta) * bin_length_in_pixels *
                         (Halide::cast<float>(i) - N / 2.f + 0.5f);
      const auto x_ij = Halide::cast<std::int32_t>(Halide::round(x + dx_ij));
      const auto y_ij = Halide::cast<std::int32_t>(Halide::round(y + dy_ij));

      // The orientation in the reoriented patch is:
      const auto ori_shifted = ori_fn - theta;
      // We have:
      // -π < ori < π
      // -π < -theta < π
      // -2π < ori - theta < 2π
      // So we make sure that the orientation is [0, 2π[ as follows:
      const auto ori = Halide::select(ori_shifted < 0,       //
                                      ori_shifted + two_pi,  //
                                      ori_shifted);          //
      const auto ori_normalized = ori / two_pi;
      const auto ori_index = ori_normalized * O;

      // For each point of the patch, i.e.:
      auto p = Vector2{};
      p(0) = x_ij + Halide::cast<float>(r.x) - x;
      p(1) = y_ij + Halide::cast<float>(r.y) - y;

      // Define the patch normalization transform.
      auto T = Matrix2{};
      T(0, 0) = Halide::cos(theta);  T(0, 1) = Halide::sin(theta);
      T(1, 0) = -Halide::sin(theta); T(1, 1) = Halide::cos(theta);
      T /= bin_length_in_pixels;

      // Calculate the coordinates of the gradient in the reoriented normalized
      // patch.
      const auto Tp = T * p;  // 1. Apply the patch normalization transform.

      // The weight of this gradient is:
      const auto weight = Halide::exp(-squared_norm(Tp) /  //
                                      (2 * Halide::pow(N / 2.f, 2)));

      auto Tp2 = Vector2{};   // 2. Find out which bin (i, j) it belongs to.
      Tp2(0) = Tp(0) + N / 2.f - 0.5f;
      Tp2(1) = Tp(1) + N / 2.f - 0.5f;

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j).
      const auto dx = Halide::abs(Halide::cast<float>(j) - Tp2(0));
      const auto dy = Halide::abs(Halide::cast<float>(i) - Tp2(1));
      // Accumulation rule based on trilinear interpolation.
      const auto wx = Halide::select(dx < 1, 1 - dx, 0);
      const auto wy = Halide::select(dy < 1, 1 - dy, 0);

      const auto o_int = Halide::cast<int>(ori_index);
      const auto o0 = Halide::clamp(                     //
          Halide::select(o_int >= O, o_int - O, o_int),  //
          0,                                             //
          O - 1);
      const auto o1 = Halide::clamp(               //
          Halide::select(o0 == O - 1, 0, o0 + 1),  //
          0,                                       //
          O - 1);

      // Now the accumulation rule is based on trilinear interpolation:
      //
      // First calculate the absolute distance to the bin (i, j, o).
      const auto dori = Halide::fract(ori_index);

      const auto wo0 = 1 - dori;
      const auto wo1 = dori;

      h(o0, ji, k) += wo0 * wx * wy * weight * mag_fn;
      h(o1, ji, k) += wo1 * wx * wy * weight * mag_fn;
    }

    //! @brief Calculate the histogram of gradients for each subpatch (i, j).
    template <typename FuncOrBuffer>
    auto accumulate_subhistogram_v3(               //
        FuncOrBuffer& h,                           //
        const Halide::Var& ji,                     //
        const Halide::Expr& k,                     //
        const Halide::Func& grad_normalized,       //
        const Halide::Func& spatial_weight) const  //
    {
      const auto i = ji / N;
      const auto j = ji - i * N;

      const auto r = Halide::RDom(                                      //
          -bin_length_in_scale_unit, 2 * bin_length_in_scale_unit + 1,  //
          -bin_length_in_scale_unit, 2 * bin_length_in_scale_unit + 1   //
      );

      const auto x = Halide::round(                      //
          (j - N / 2 + 0.5f) * bin_length_in_scale_unit  //
      );
      const auto y = Halide::round(                      //
          (i - N / 2 + 0.5f) * bin_length_in_scale_unit  //
      );

      const auto xi = Halide::cast<std::int32_t>(x);
      const auto yi = Halide::cast<std::int32_t>(y);

      const auto mag = grad_normalized(xi + r.x, yi + r.y, k)[0];
      const auto ori = grad_normalized(xi + r.x, yi + r.y, k)[1];

      // Trilinear interpolation.
      //
      // Fetch the precomputed spatial weights.
      const auto wx_times_wy = spatial_weight(r.x, r.y);

      // Orientation weights.
      const auto o_int = Halide::cast<int>(Halide::floor(ori));
      const auto o0 = Halide::clamp(                     //
          Halide::select(o_int >= O, o_int - O, o_int),  //
          0, O - 1);
      const auto o1 = Halide::clamp(               //
          Halide::select(o0 == O - 1, 0, o0 + 1),  //
          0, O - 1);
      const auto dori = Halide::fract(ori);
      // Thus:
      const auto wo0 = 1 - dori;
      const auto wo1 = dori;

      // Accumulation rule.
      h(o0, ji, k) += wo0 * wx_times_wy * mag;
      h(o1, ji, k) += wo1 * wx_times_wy * mag;
    }

    auto normalize_histogram(const Halide::Func& h,  //
                             const Halide::Var& o,   //
                             const Halide::Var& ji,  //
                             const Halide::Var& k)   //
    {
      const auto r = Halide::RDom(0, O, 0, N * N);

      // 1.   Invariance to contrast changes.
      {
        contrast_norm(k) = Halide::sqrt(                 //
            Halide::sum(Halide::pow(h(r.x, r.y, k), 2))  //
        );
        // contrast_norm.compute_at(h, k);
        contrast_norm.compute_root();

        hist_contrast_invariant(o, ji, k) = h(o, ji, k) / contrast_norm(k);
        // hist_contrast_invariant.compute_at(contrast_norm, k);
        hist_contrast_invariant.compute_root();
      }

      // 2.   Nonlinear illumination changes.
      // 2.a) Clamp histogram values.
      {
        hist_clamped(o, ji, k) = Halide::min(   //
            hist_contrast_invariant(o, ji, k),  //
            max_bin_value                       //
        );
        //hist_clamped.compute_at(hist_contrast_invariant, k);
        hist_clamped.compute_root();
      }

      // 2.b) Normalize.
      {
        illumination_norm(k) = Halide::sqrt(                        //
            Halide::sum(Halide::pow(hist_clamped(r.x, r.y, k), 2))  //
        );
        // illumination_norm.compute_at(hist_clamped, k);
        illumination_norm.compute_root();

        hist_illumination_invariant(o, ji, k) = hist_clamped(o, ji, k) /  //
                                                illumination_norm(k);
        // hist_illumination_invariant.compute_at(illumination_norm, k);
        hist_illumination_invariant.compute_root();
      }
    }
  };

}  // namespace DO::Shakti::HalideBackend
