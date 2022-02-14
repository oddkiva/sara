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

#include "Chessboard/EdgeFusion.hpp"
#include "Chessboard/SaddleDetection.hpp"


namespace sara = DO::Sara;
using sara::operator""_deg;


// TODO:
// 1. Group adjacent edges if their mean gradients are aligned.
//    This will merge adjacent edges that were split too early.
//    THIS IS ✔️
//
// 2. Group aligned edges with alternating gradient orientation.
//    This will group edge chessboard corners with alternating colors.
//    Using the mean gradient signature and std dev.
//    THIS IS ❌
//
// 3. Group perpendicular long edges.
//    HOW?
//    Exploit the mean gradient signature and the std dev.
//    THIS IS ❌
//
// 4. Identify saddle points that are chessboard corners.
//    Label them with an index (i, j).
//    HOW?
//    This is ❌
//
// 5. Refine chessboard corners:
//    Optimizing the Determinant of Hessian:
//    It must be negative and have the largest possible absolute value.
//    This should be a locally convex problem
//    This is ❌


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


auto edge_signature(const sara::ImageView<sara::Rgb8>& color,
                    const sara::ImageView<float>& gradient_mag,
                    const sara::ImageView<float>& gradient_ori,
                    const std::vector<Eigen::Vector2i>& edge,  //
                    float delta = 1, int width = 3)
{
  auto darks = std::vector<sara::Rgb64f>{};
  auto brights = std::vector<sara::Rgb64f>{};
  for (auto s = 1; s <= width; ++s)
  {
    for (const auto& e : edge)
    {
      const auto ori = gradient_ori(e);
      const auto n = Eigen::Vector2d(std::cos(ori), std::sin(ori));

      const Eigen::Vector2d b = e.cast<double>() + s * delta * n;
      const Eigen::Vector2d d = e.cast<double>() - s * delta * n;

      if (0 <= d.x() && d.x() < color.width() &&  //
          0 <= d.y() && d.y() < color.height())
        darks.push_back(sara::interpolate(color, d));

      if (0 <= b.x() && b.x() < color.width() &&  //
          0 <= b.y() && b.y() < color.height())
        brights.push_back(sara::interpolate(color, b));
    }
  }

  Eigen::Vector2f mean_gradient = Eigen::Vector2f::Zero();
  mean_gradient = std::accumulate(  //
      edge.begin(), edge.end(), mean_gradient,
      [&gradient_mag, &gradient_ori](const auto& g,
                                     const auto& e) -> Eigen::Vector2f {
        if (gradient_mag(e) < 1e-3f)
          return g;
        const auto ori = gradient_ori(e);
        const auto n = Eigen::Vector2f(std::cos(ori), std::sin(ori));

        return g + n;
      });
  mean_gradient /= static_cast<float>(edge.size());

  return std::make_tuple(darks, brights, mean_gradient);
}


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

      const auto svd = s.hessian.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
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
