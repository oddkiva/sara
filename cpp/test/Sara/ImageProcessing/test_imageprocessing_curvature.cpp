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

#define BOOST_TEST_MODULE "ImageProcessing/Color Statistics"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/ImageProcessing/Curvature.hpp>


using namespace std;
using namespace DO::Sara;


auto radial_distance(Image<float>& phi,              //
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

BOOST_AUTO_TEST_CASE(test_mean_curvature)
{
  const auto w = 10;
  const auto h = 10;
  auto phi = Image<float, 2>(w, h);
  radial_distance(phi, Eigen::Vector2f(w, h) / 2, w / 2.);

  auto grad_phi = gradient(phi);
  auto grad_phi_x = grad_phi.cwise_transform([](const auto& v) { return v.x(); });
  auto grad_phi_y = grad_phi.cwise_transform([](const auto& v) { return v.y(); });
  auto grad_phi_norm = grad_phi.cwise_transform([](const auto& v) { return v.norm(); });
  std::cout << "grad_x =\n" << grad_phi_x.matrix() << std::endl;
  std::cout << "grad_y =\n" << grad_phi_y.matrix() << std::endl;
  std::cout << "grad_norm =\n" << grad_phi_norm.matrix() << std::endl;

  auto curvature = Image<float>{w, h};
  for (auto y = 0; y < h; ++y)
    for (auto x = 0; x < w; ++x)
      curvature(x, y) = mean_curvature(phi, {x, y}, 1e-2f);


  std::cout << "Curvature =\n" << curvature.matrix() << std::endl;

  auto radius = Image<float>{w, h};
  for (auto y = 0; y < h; ++y)
    for (auto x = 0; x < w; ++x)
      radius(x, y) = 1 / curvature(x, y);
}


