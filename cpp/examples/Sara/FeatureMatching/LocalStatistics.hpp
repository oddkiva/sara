#pragma once

#include <DO/Sara/Core/Tensor.hpp>


namespace DO::Sara {

  struct LocalPhotometricStatistics
  {
    int patch_radius = 4;

    std::vector<Eigen::Vector2i> point_set;

    // Dense maps.
    Tensor_<float, 3> rgb;
    Tensor_<float, 3> rgb_mean;
    Tensor_<float, 3> rgb_unnormalized_std_dev;

    auto resize(int width, int height)
    {
      rgb.resize({3, height, width});
      rgb_mean.resize({3, height, width});
      rgb_unnormalized_std_dev.resize({3, height, width});
    }

    auto swap(LocalPhotometricStatistics& other)
    {
      point_set.swap(other.point_set);
      rgb.swap(other.rgb);
      rgb_mean.swap(other.rgb_mean);
      rgb_unnormalized_std_dev.swap(other.rgb_unnormalized_std_dev);
    }

    auto update_point_set(const std::vector<Eigen::Vector2i>& pts)
    {
      const auto& r = patch_radius;
      const auto w = rgb.size(2);
      const auto h = rgb.size(1);

      point_set.clear();
      point_set.reserve(pts.size());

      for (const auto& p : pts)
        if (r <= p.x() && p.x() < w - r &&  //
            r <= p.y() && p.y() < h - r)
          point_set.emplace_back(p);
    }

    auto update_image(const ImageView<Rgb8>& image)
    {
      auto r_ptr = rgb[0].data();
      auto g_ptr = rgb[1].data();
      auto b_ptr = rgb[2].data();
      auto rgb_ptr = image.data();

      SARA_CHECK(image.width());
      SARA_CHECK(image.height());
      SARA_CHECK(image.size());

#pragma omp parallel for
      for (auto xy = 0u; xy < image.size(); ++xy)
      {
        const auto rgb = *(rgb_ptr + xy);
        *(r_ptr + xy) = float(rgb[0]) / 255;
        *(g_ptr + xy) = float(rgb[1]) / 255;
        *(b_ptr + xy) = float(rgb[2]) / 255;
      }

#define INSPECT
#ifdef INSPECT
      for (auto c = 0; c < 3; ++c)
      {
        display(image_view(rgb[c]));
        get_key();
      }
#endif
    }

    auto calculate_mean()
    {
      // Reset the mean.
      for (auto i = 0; i < 3; ++i)
      {
        auto plane = rgb_mean[i];
        plane.flat_array().setZero();
      }

      const auto& r = patch_radius;
      const auto area = std::pow(2 * r, 2);

      auto calculate_mean_at = [&](const TensorView_<float, 2>& plane, int x,
                                   int y) {
        auto mean = float{};

        const auto xmin = x - r;
        const auto xmax = x + r;
        const auto ymin = y - r;
        const auto ymax = y + r;
        for (auto v = ymin; v < ymax; ++v)
          for (auto u = xmin; u < xmax; ++u)
            mean += plane(v, u);
        mean /= area;

        return mean;
      };

      // Loop through the points.
      for (auto i = 0; i < 3; ++i)
      {
        auto f = rgb[i];
        auto f_mean = rgb_mean[i];

#pragma omp parallel for
        for (auto i = 0u; i < point_set.size(); ++i)
        {
          const auto& p = point_set[i];
          f_mean(p.y(), p.x()) = calculate_mean_at(f, p.x(), p.y());
        }
      }

#ifdef INSPECT
      // TODO: check each plane on the display.
      auto rgb_mean_transposed = rgb_mean.transpose({1, 2, 0});
      auto rgb32f_view = ImageView<Rgb32f>(
          reinterpret_cast<Rgb32f*>(rgb_mean_transposed.data()),
          {rgb_mean_transposed.size(1), rgb_mean_transposed.size(0)});
      display(rgb32f_view);
#endif
    }

    auto calculate_unnormalized_std_dev()
    {
      // Reset the second order moment.
      for (auto i = 0; i < 3; ++i)
      {
        auto plane = rgb_unnormalized_std_dev[i];
        plane.flat_array().setZero();
      }

      const auto& r = patch_radius;
      auto calculate_unnormalized_std_dev_at =
          [&](const TensorView_<float, 2>& plane, float mean, int x, int y) {
            auto sigma = float{};

            const auto xmin = x - r;
            const auto xmax = x + r;
            const auto ymin = y - r;
            const auto ymax = y + r;
            for (auto v = ymin; v < ymax; ++v)
              for (auto u = xmin; u < xmax; ++u)
                sigma += std::pow(plane(v, u) - mean, 2);

            return sigma;
          };

      // Loop through the points.
      for (auto i = 0; i < 3; ++i)
      {
        auto f = rgb[i];
        auto f_mean = rgb_mean[i];
        auto f_sigma = rgb_unnormalized_std_dev[i];

#pragma omp parallel for
        for (auto i = 0u; i < point_set.size(); ++i)
        {
          const auto& p = point_set[i];
          const auto f_mean_xy = f_mean(p.y(), p.x());
          f_sigma(p.y(), p.x()) =
              calculate_unnormalized_std_dev_at(f,          //
                                                f_mean_xy,  //
                                                p.x(), p.y());
        }
      }

#ifdef INSPECT
      // TODO: check each plane on the display.
      for (auto i = 0; i < 3; ++i)
      {
        display(image_view(rgb_unnormalized_std_dev[i]));
        get_key();
      }
#endif
    }
  };

}  // namespace DO::Sara
