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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include "Chessboard/SaddleDetection.hpp"


namespace sara = DO::Sara;


namespace DO::Sara {

  auto weight_mask(const std::vector<int>& radius)
  {
  }

  auto filter_saddle_points(const ImageView<float>& image,
                            const ImageView<float>& gradient_ori,
                            const ImageView<float>& gradient_mag,
                            const std::vector<SaddlePoint>& saddle_points)
  {
    static constexpr auto num_circle_points = 32;
    static constexpr auto num_bins = 32;
    static constexpr auto crossing_threshold = 3;
    static constexpr auto need_crossing = 4;
    static constexpr auto need_mode = 2;

    static constexpr auto radius = 5.;

    const auto w = image.width();
    const auto h = image.height();

    auto ori_vectors = std::array<Eigen::Vector2d, num_circle_points>{};
    for (auto i = 0; i < num_circle_points; ++i)
    {
      static const auto pi = static_cast<double>(M_PI);
      const auto angle = i * 2 * pi / (num_circle_points - 1);
      ori_vectors[i] << std::cos(angle), std::sin(angle);
    }

    auto saddle_points_filtered = std::vector<SaddlePoint>{};

    for (const auto& s : saddle_points)
    {
      // Sample the image intensity along the circle.
      auto circle_intensities = Eigen::Array<float, num_circle_points, 1>{};
      for (auto n = 0; n < num_circle_points; ++n)
      {
        const Eigen::Vector2d circle_point_n =
            s.p.cast<double>() + radius * ori_vectors[n];

        // Get the interpolated intensity value.
        circle_intensities[n] = interpolate(image, circle_point_n);
      }

      // Rescale the intensity values.
      const auto min_intensity = circle_intensities.minCoeff();
      const auto max_intensity = circle_intensities.minCoeff();
      const auto mid_intensity = (max_intensity + min_intensity) * 0.5f;
      circle_intensities = circle_intensities - mid_intensity;

      // ANALYSIS ON IMAGE INTENSITY:
      // Count the number of zero-crossings: there must be 4 zero-crossings
      // because of the chessboard pattern.
      auto num_crossings = 0;
      for (auto n = 0; n < num_circle_points; ++n)
      {
        // A zero-crossing is characterized by a negative sign between
        // consecutive intensity values.
        const auto& a = circle_intensities[n];
        const auto& b = circle_intensities[(n + 1) % num_circle_points];
        if (a * b < 0)
          ++num_crossings;
      }

      auto image_patch = safe_crop(gradient_mag, s.p, std::round(radius));

      // TODO: apply the mask.
      // image_patch.array() *= mask[r].array();

      // Calculate the half of the max intensity value in the image patch.
      const auto half_max_patch_intensity =
          image_patch.flat_array().maxCoeff() * 0.5f;

      // Clamp the intensity values.
      image_patch.cwise_transform_inplace(
          [half_max_patch_intensity](auto& val) {
            val = val < half_max_patch_intensity ? 0 : val;
          });

      // ANALYSIS ON THE IMAGE GRADIENTS.
      //
      // Count the number gradient orientation peaks in the image patch: there
      // must be only two peaks in the gradient absolute orientation because of
      // the chessboard pattern.
      auto ori_hist = Eigen::Array<float, num_bins, 1>{};
      ori_hist.setZero();
      auto r = int{};
      for (auto v = -r; v <= r; ++v)
      {
        for (auto u = -r; u <= r; ++u)
        {
          const auto& angle = gradient_ori(s.p.x() + u, s.p.y() + v);
          const auto angle_normalized =
              static_cast<float>(std::abs(angle) / M_PI) * num_bins;
          auto angle_int = float{};
          const auto angle_frac = std::modf(angle_normalized, &angle_int);
          const auto angle_weight = 1 - angle_frac;

          auto angle_bin_0 = static_cast<int>(angle_int);
          if (angle_bin_0 == num_bins)
            angle_bin_0 = 0;
          const auto angle_bin_1 = (angle_bin_0 + 1) % num_bins;

          ori_hist[angle_bin_0] += angle_weight;
          ori_hist[angle_bin_1] += (1 - angle_weight);
        }
      }

      // Find the mean shift modes, or simply perform like David Lowe, smoothe
      // the orientation histogram and find the peaks.

      // TODO: keep only the candidate saddle points where there are 4
      // zero-crossings and only two mode gradient modes.
    }

    return saddle_points_filtered;
  }

}  // namespace DO::Sara

int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  if (argc < 2)
    return 1;
  const auto folder = std::string{argv[1]};

  static constexpr auto sigma = 1.6f;
  static constexpr auto nms_radius = 10;
  static constexpr float adaptive_thres = 0.05f;

  for (auto i = 0; i <= 1790; i += 10)
  {
    const auto image_filepath = folder + "/" + std::to_string(i) + ".png";

    auto image = sara::imread<sara::Rgb8>(image_filepath);
    const auto image_gray = image.convert<float>();
    const auto image_blurred = image_gray.compute<sara::Gaussian>(sigma);

    if (sara::active_window() == nullptr)
    {
      sara::create_window(image.sizes());
      sara::set_antialiasing();
    }

    // Calculate the first derivative.
    const auto hessian = image_blurred.compute<sara::Hessian>();

    // Chessboard corners are saddle points of the image, which are
    // characterized by the property det(H(x, y)) < 0.
    const auto det_of_hessian = hessian.compute<sara::Determinant>();

    // Adaptive thresholding.
    const auto thres = det_of_hessian.flat_array().minCoeff() * adaptive_thres;
    auto saddle_points = extract_saddle_points(det_of_hessian, hessian, thres);

    // Non-maxima suppression.
    nms(saddle_points, image.sizes(), nms_radius);

    for (const auto& s : saddle_points)
    {
      sara::fill_circle(image, s.p.x(), s.p.y(), 5, sara::Red8);

      const auto svd =
          s.hessian.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Eigen::Vector2f S = svd.singularValues();
      const auto& axes = svd.matrixU();

      const auto a = Eigen::Vector2f(s.p.x(), s.p.y());
      static constexpr auto radius = 20.f;
      const Eigen::Vector2f b = a + radius * axes.col(0);
      const Eigen::Vector2f c = a + radius * axes.col(1);

      sara::draw_arrow(image, a, b, sara::Cyan8, 2);
      sara::draw_arrow(image, a, c, sara::Cyan8, 2);
    }

    sara::display(image);
    sara::get_key();
  }

  return 0;
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
