#pragma once

#include <DO/Sara/Core.hpp>


template <Eigen::Index N>
void compute_orientation_histogram(
    Eigen::Array<float, N, 1>& orientation_histogram,
    const DO::Sara::ImageView<float>& grad_f_norm,
    const DO::Sara::ImageView<float>& grad_f_ori,     //
    const float x, const float y, const float s,  //
    const float patch_truncation_factor = 3.f,    //
    const float blur_factor = 1.5f)
{
  // Weighted histogram of gradients.
  orientation_histogram.setZero();

  // Rounding of the coordinates.
  static constexpr auto int_round = [](const float x) {
    return static_cast<int>(std::round(x));
  };
  auto rounded_x = int_round(x);
  auto rounded_y = int_round(y);

  // std deviation of the gaussian weight (cf. [Lowe, IJCV 2004])
  auto sigma = s * blur_factor;

  // Patch radius on which the histogram of gradients is performed.
  auto patch_radius = int_round(sigma * patch_truncation_factor);
  const auto w = grad_f_norm.width();
  const auto h = grad_f_norm.height();

  const auto one_over_two_sigma_square = 1 / (2.f * sigma * sigma);

  // Accumulate the histogram of orientations.
  for (auto v = -patch_radius; v <= patch_radius; ++v)
  {
    for (auto u = -patch_radius; u <= patch_radius; ++u)
    {
      if (rounded_x + u < 0 || rounded_x + u >= w ||  //
          rounded_y + v < 0 || rounded_y + v >= h)
        continue;

      const auto mag = grad_f_norm(rounded_x + u, rounded_y + v);
      auto ori = grad_f_ori(rounded_x + u, rounded_y + v);

      // ori is in \f$]-\pi, \pi]\f$, so translate ori by \f$2*\pi\f$ if it is
      // negative.
      static constexpr auto two_pi = static_cast<float>(2 * M_PI);
      static constexpr auto normalization_factor = N / two_pi;

      ori = ori < 0 ? ori + two_pi : ori;
      auto bin_value = ori * normalization_factor;
      auto bin_int = float{};
      const auto bin_frac = std::modf(bin_value, &bin_int);

      const auto bin_i0 = static_cast<int>(bin_int) % N;
      const auto bin_i1 = (bin_i0 + 1) % N;

      // Distribute on two adjacent bins with linear interpolation.
      const auto w0 = (1 - bin_frac);
      const auto w1 = bin_frac;

      // Give more emphasis to gradient orientations that lie closer to the
      // keypoint location.
      // Also give more emphasis to gradient with large magnitude.
      const auto w = exp(-(u * u + v * v) * one_over_two_sigma_square) * mag;
      orientation_histogram(bin_i0) += w0 * w;
      orientation_histogram(bin_i1) += w1 * w;
    }
  }
}
