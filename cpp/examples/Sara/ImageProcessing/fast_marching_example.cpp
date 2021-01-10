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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/FastMarching.hpp>

#include "Utilities.hpp"

#include <iostream>
#include <numeric>
#include <set>


namespace sara = DO::Sara;

auto radial_distance(sara::Image<float>& phi, const Eigen::Vector2f& center)
{
  for (auto y = 0; y < phi.height(); ++y)
  {
    for (auto x = 0; x < phi.width(); ++x)
    {
      const auto xy = Eigen::Vector2f(x, y);
      phi(x, y) = (xy - center).norm();
    }
  }
}


enum State : std::uint8_t
{
  Alive = 0,
  Trial = 1,
  Far = 2,
  Forbidden = 3
};

struct CoordsValue
{
  Eigen::Vector2i coords;
  float val;

  inline auto operator<(const CoordsValue& other) const
  {
    return val < other.val;
  }
};



GRAPHICS_MAIN()
{
#define REAL_IMAGE
#ifdef REAL_IMAGE
  const auto image = sara::imread<float>(                       //
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/stinkbug.png"        //
#else
      "/home/david/GitLab/DO-CV/sara/data/stinkbug.png"        //
#endif
  );

  constexpr auto sigma = 3.f;
  const auto image_blurred = sara::gaussian(image, sigma);
  const auto laplacian = sara::laplacian(image_blurred);

  const auto grad = sara::gradient(image_blurred);
  const auto grad_mag =
      grad.cwise_transform([](const auto& v) { return v.squaredNorm(); });

  static_assert(std::is_same_v<decltype(image_blurred), decltype(grad_mag)>);

  // Create the speed function from the gradient magnitude.
  auto speed_times_dt = grad_mag.cwise_transform(
      [](const auto& v) { return std::exp(-v); }  //
  );

  // Extract the zero level set.
  const auto zeros = sara::extract_zero_level_set(laplacian);

  sara::create_window(image.sizes());
  sara::display(sara::color_rescale(laplacian));
#else
  const auto w = 512;
  const auto h = 512;
  auto image = sara::Image<float, 2>{w, h};
  radial_distance(image, Eigen::Vector2f(w, h) / 2);

  // const auto grad = sara::gradient(image);
  // const auto grad_mag =
  //     grad.cwise_transform([](const auto& v) { return v.norm(); });

  auto speed_times_dt = sara::Image<float, 2>{w, h};
  speed_times_dt.flat_array().fill(1);

  const auto zeros = std::vector{Eigen::Vector2i(w/2, h/2)};

  sara::create_window(image.sizes());
  sara::display(sara::color_rescale(image));
#endif

  for (const auto& p : zeros)
    sara::draw_point(p.x(), p.y(), sara::Red8);

  // Fast marching.
  sara::tic();
  sara::FastMarching<float, 2> fm{speed_times_dt, 1};
  fm.initialize_alive_points(zeros);
  fm.run();
  sara::toc("Fast Marching 2D");

  auto distances = fm._distances;
  for (auto y = 0; y < distances.height(); ++y)
    for (auto x = 0; x < distances.width(); ++x)
      if (distances(x, y) == std::numeric_limits<float>::max())
        distances(x, y) = 0;

  auto d = distances.flat_array();
  const auto dmax = d.maxCoeff();
  d /= dmax;

  sara::display(distances);
  sara::get_key();

  return 0;
}
