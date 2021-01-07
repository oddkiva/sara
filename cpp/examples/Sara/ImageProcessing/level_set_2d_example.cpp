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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/LevelSets/TimeIntegrators.hpp>

#include <iostream>


namespace sara = DO::Sara;


auto spherical_distance(sara::Image<float>& phi,        //
                        const Eigen::Vector2f& center,  //
                        float radius)
{
  for (auto y = 0; y < phi.height(); ++y)
  {
    for (auto x = 0; x < phi.width(); ++x)
    {
      const auto xy = Eigen::Vector2f(x, y);
      phi(x, y) = (xy - center).norm() - radius;
    }
  }
}

auto manhattan_distance(sara::Image<float>& phi,        //
                        const Eigen::Vector2f& center,  //
                        const Eigen::Vector2f& ab)
{
  for (auto y = 0; y < phi.height(); ++y)
  {
    for (auto x = 0; x < phi.width(); ++x)
    {
      const auto xy = Eigen::Vector2f(x, y);
      const auto d = ((xy - center).cwiseAbs() - ab).eval();
      phi(x, y) = d.x() > 0 && d.y() > 0 ? d.norm() : d.maxCoeff();
    }
  }
}

auto extract_zero_level_set(const sara::Image<float>& phi)
{
  auto zeros = std::vector<Eigen::Vector2i>{};
  zeros.reserve(phi.sizes().maxCoeff());

  for (auto y = 1; y < phi.height() - 1; ++y)
  {
    for (auto x = 1; x < phi.width() - 1; ++x)
    {
      const auto sx = phi(x - 1, y) * phi(x + 1, y);
      const auto sy = phi(x, y - 1) * phi(x, y + 1);
      if (sx < 0 || sy < 0)
        zeros.emplace_back(x, y);
    }
  }

  return zeros;
}


void level_set()
{
  // Load level set function
  const auto w = 128;
  const auto h = 96;
  const auto scale = 4;
  auto phi = sara::Image<float, 2>(w, h);
  // spherical_distance(phi, Eigen::Vector2f(w, h) / 2, w / 5.);
  manhattan_distance(phi, Eigen::Vector2f(w, h) / 2, Eigen::Vector2f(w, h) / 4);

  // For display purposes.
  sara::create_window(scale * w, scale * h);

  auto phi_enlarged = sara::enlarge(phi, scale);
  auto image = phi_enlarged;
  const auto max_abs_value = image.flat_array().abs().maxCoeff();
  image.flat_array() /= max_abs_value;
  image.flat_array() += 0.5;
  image = sara::color_rescale(image, 0.f, 1.f);
  sara::display(image);

  const auto zeros = extract_zero_level_set(phi_enlarged);
  SARA_CHECK(zeros.size());
  for (const auto& p : zeros)
    sara::draw_point(p.x(), p.y(), sara::Red8);
  sara::get_key();


GRAPHICS_MAIN()
{
  level_set();
  return 0;
}
