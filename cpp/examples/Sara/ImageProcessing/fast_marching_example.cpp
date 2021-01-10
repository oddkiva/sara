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


auto fast_marching_2d() -> void
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
  auto speed_times_dt = sara::Image<float, 2>{w, h};
  speed_times_dt.flat_array().fill(1);

  const auto zeros = std::vector{Eigen::Vector2i(w/2, h/2)};

  sara::create_window(image.sizes());
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
}


auto fast_marching_3d() -> void
{
  const auto w = 128;
  const auto h = 128;
  const auto d = 128;

  auto speed_times_dt = sara::Image<float, 3>{w, h, d};
  speed_times_dt.flat_array().fill(1);

  const auto zeros = std::vector{Eigen::Vector3i(w/2, h/2, d/2)};

  // Fast marching.
  sara::tic();
  auto fm = sara::FastMarching<float, 3>{speed_times_dt};
  fm.initialize_alive_points(zeros);
  fm.run();
  sara::toc("Fast Marching 3D");

  auto distances = fm._distances;
  for (auto z = 0; z < distances.depth(); ++z)
    for (auto y = 0; y < distances.height(); ++y)
      for (auto x = 0; x < distances.width(); ++x)
        if (distances(x, y, z) == std::numeric_limits<float>::max())
          distances(x, y, z) = 0;

  auto dist = distances.flat_array();
  const auto dmax = dist.maxCoeff();
  dist /= dmax;

  auto tensor = sara::tensor_view(distances);

  auto scale = 2;
  sara::create_window(scale * distances.width(), scale * distances.height());
  for (auto z = fm._margin.z(); z < d - fm._margin.z(); ++z)
  {
    const auto imview = sara::image_view(tensor[z]);
    sara::display(imview, 0, 0, scale);
    sara::get_key();
  }
}


GRAPHICS_MAIN()
{
  fast_marching_2d();
  // fast_marching_3d();
  return 0;
}
