#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

// ========================================================================== //
// Utilities.
// ========================================================================== //
template <typename T>
auto find_peaks(
    const Map<const Eigen::Array<T, Eigen::Dynamic, 1>>& circular_data,  //
    Map<Eigen::Array<std::uint8_t, Eigen::Dynamic, 1>>& peaks,           //
    T peak_ratio_thres = static_cast<T>(0.8))                            //
    -> void
{
  if (circular_data.size() == 0)
    return;

  const auto& max = circular_data.maxCoeff();
  const auto& N = circular_data.size();

  for (auto i = Eigen::Index{}; i < N; ++i)
  {
    const auto prev = i > 0 ? i - 1 : N;
    const auto next = i < N - 1 ? i + 1 : 0;
    peaks[i] = circular_data(i) >= peak_ratio_thres * max &&
               circular_data(i) > circular_data(prev) &&
               circular_data(i) > circular_data(next);
  }
}

template <typename T>
auto peak_residual(
    const Map<const Eigen::Array<T, Eigen::Dynamic, 1>>& circular_data,  //
    Eigen::Index i)                                                      //
    -> T
{
  const auto& N = circular_data.size();

  const auto prev = i > 0 ? i - 1 : N;
  const auto next = i < N - 1 ? i + 1 : 0;

  const auto& y0 = circular_data(prev);
  const auto& y1 = circular_data(i);
  const auto& y2 = circular_data(next);

  const auto fprime = (y2 - y0) / 2.f;
  const auto fsecond = y0 - 2.f * y1 + y2;

  const auto h = -fprime / fsecond;

  return T(i) + T(0.5) + h;
}


// ========================================================================== //
// Operation on tensors.
// ========================================================================== //
auto orientation_histograms(const ImageView<std::uint8_t>& edge_map,  //
                            const ImageView<float>& orientation_map,  //
                            int num_bins = 18, int radius = 5)
{
  auto histograms =
      Tensor_<float, 3>{edge_map.height(), edge_map.width(), num_bins};
  histograms.flat_array().fill(0);

  const auto& r = radius;

#pragma omp parallel for
  for (auto y = 0; y < edge_map.height(); ++y)
  {
    for (auto x = 0; x < edge_map.width(); ++x)
    {
      if (edge_map(x, y) == 0)
        continue;

      const auto& label = edge_map(x, y);

      // Loop over the pixels in the patch centered in p.
      for (auto v = -r; v <= r; ++v)
      {
        for (auto u = -r; u <= r; ++u)
        {
          // Boundary conditions.
          const auto n = Eigen::Vector2i{x + u, y + v};
          if (n.x() < 0 || n.x() >= edge_map.width() ||  //
              n.y() < 0 || n.y() >= edge_map.height())
            continue;

          // Only consider neighbors with the same label.
          if  (edge_map(n) != label)
            continue;

          const auto& orientation = std::abs(orientation_map(n) - orientation_map(x, y));

          auto ori_0O = static_cast<float>(orientation / (2 * M_PI) * num_bins);
          if (ori_0O >= num_bins)
            ori_0O -= num_bins;
          auto ori_intf = decltype(ori_0O){};
          const auto ori_frac = std::modf(ori_0O, &ori_intf);

          auto ori_0 = int(ori_intf);
          auto ori_1 = ori_0 + 1;
          if (ori_1 == num_bins)
            ori_1 = 0;

          histograms(y, x, ori_0) += (1 - ori_frac);
          histograms(y, x, ori_1) += ori_frac;
        }
      }
    }
  }

  return histograms;
}

auto peaks(const TensorView_<float, 3>& histograms,
           float peak_ratio_thres = 0.8f)
{
  auto peaks = Tensor_<std::uint8_t, 3>{histograms.sizes()};

#pragma omp parallel for
  for (auto y = 0; y < histograms.size(0); ++y)
  {
    for (auto x = 0; x < histograms.size(1); ++x)
    {
      const auto& h = histograms[y][x].flat_array();
      auto p = peaks[y][x].flat_array();
      find_peaks(h, p, peak_ratio_thres);
    }
  }

  return peaks;
}

auto peak_counts(const TensorView_<std::uint8_t, 3>& peaks)
{
  const Eigen::Vector2i num_peaks_sizes = peaks.sizes().head(2);
  auto peak_counts = Tensor_<std::uint8_t, 2>{num_peaks_sizes};

#pragma omp parallel for
  for (auto y = 0; y < peak_counts.size(0); ++y)
    for (auto x = 0; x < peak_counts.size(1); ++x)
      peak_counts(y, x) = peaks[y][x].flat_array().count();

  return peak_counts;
}

auto peak_residuals(const TensorView_<float, 3>& circular_data,
                    const TensorView_<std::uint8_t, 3>& peak_indices)
{
  auto peak_residuals = Tensor_<float, 3>{circular_data.sizes()};

#pragma omp parallel for
  for (auto y = 0; y < circular_data.size(0); ++y)
  {
    for (auto x = 0; x < circular_data.size(1); ++x)
    {
      const auto& h = circular_data[y][x].flat_array();
      auto p = peak_indices[y][x].flat_array();

      for (auto i = 0; i < p.size(); ++i)
        peak_residuals(y, x, i) = p[i] == 0 ? 0 : peak_residual(h, i);
    }
  }

  return peak_residuals;
}

}
