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

#pragma once


namespace DO::Sara {

  struct SymGradientHistogramExtractor
  {
    // ANALYSIS ON THE IMAGE GRADIENTS.
    //
    // Count the number gradient orientation peaks in the image patch: there
    // must be only two peaks in the gradient absolute orientation because of
    // the chessboard pattern.
    auto operator()(const sara::ImageView<float>& grad_mag,
                    const sara::ImageView<float>& grad_ori,
                    const Eigen::Vector2i& p) const -> Eigen::ArrayXf
    {
      // const auto image_patch = safe_crop(grad_mag, p, std::round(_radius));

      // // Calculate the half of the max intensity value in the image patch.
      // const auto half_max_patch_intensity =
      //     image_patch.flat_array().maxCoeff() * 0.5f;

      auto ori_hist = Eigen::ArrayXf{_num_bins};
      ori_hist.setZero();

      const auto& w = grad_mag.width();
      const auto& h = grad_mag.height();

      for (auto v = -_radius; v <= _radius; ++v)
      {
        for (auto u = -_radius; u <= _radius; ++u)
        {
          const Eigen::Vector2i p_uv = p + Eigen::Vector2i(u, v);
          const auto in_image_domain = p_uv.x() >= 0 && p_uv.x() < w &&  //
                                       p_uv.y() >= 0 && p_uv.y() < h;
          if (!in_image_domain)
            continue;

          const auto& mag = grad_mag(p_uv);

          static constexpr auto pif = static_cast<float>(M_PI);
          auto angle = grad_ori(p_uv);
          if (angle < 0)
            angle += pif;

          const auto angle_normalized = angle / pif * _num_bins;
          auto angle_int = float{};
          const auto angle_frac = std::modf(angle_normalized, &angle_int);

          const auto angle_weight = 1 - angle_frac;

          auto angle_bin_0 = static_cast<int>(angle_int);
          if (angle_bin_0 == _num_bins)
            angle_bin_0 = 0;
          const auto angle_bin_1 = (angle_bin_0 + 1) % _num_bins;

          ori_hist[angle_bin_0] += angle_weight * mag;
          ori_hist[angle_bin_1] += (1 - angle_weight) * mag;
        }
      }

      // Smooth in place: it works.
      static constexpr auto num_iters = 8;
      static constexpr auto one_third = 1 / 3.f;
      for (int iter = 0; iter < num_iters; ++iter)
      {
        const auto first = ori_hist(0);
        auto prev = ori_hist(_num_bins - 1);
        for (int i = 0; i < _num_bins - 1; ++i)
        {
          const auto val = (prev + ori_hist(i) + ori_hist(i + 1)) * one_third;
          prev = ori_hist(i);
          ori_hist(i) = val;
        }
        ori_hist(_num_bins - 1) =
            (prev + ori_hist(_num_bins - 1) + first) * one_third;
      }

      return ori_hist;
    }

    int _num_bins = 36;
    int _radius = 11;
  };

  inline auto find_peaks(const Eigen::ArrayXf& orientation_histogram,
                         float peak_ratio_thres = 0.5f) -> std::vector<int>
  {
    const auto max = orientation_histogram.maxCoeff();
    std::vector<int> orientation_peaks;

    const auto n = orientation_histogram.size();
    for (int i = 0; i < n; ++i)
      if (orientation_histogram(i) >= peak_ratio_thres * max &&  //
          orientation_histogram(i) > orientation_histogram((i - 1 + n) % n) &&
          orientation_histogram(i) > orientation_histogram((i + 1) % n))
        orientation_peaks.push_back(i);
    return orientation_peaks;
  }


}  // namespace DO::Sara
