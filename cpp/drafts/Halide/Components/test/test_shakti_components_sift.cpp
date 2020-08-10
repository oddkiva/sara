// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Halide Backend/SIFT components"

#include <iomanip>

#include <boost/test/unit_test.hpp>

#include <Eigen/Dense>

#include <DO/Sara/ImageProcessing.hpp>

#include <drafts/Halide/Components/SIFT.hpp>
#include <drafts/Halide/Utilities.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;

using namespace Halide;


auto make_corner_image()
{
  auto image = sara::Image<float>{10, 10};
  image.flat_array().fill(0);

  for (auto y = image.height() / 2; y < image.height(); ++y)
    for (auto x = 0; x < image.width() / 2; ++x)
      image(x, y) = 1;

  return image;
}


BOOST_AUTO_TEST_CASE(test_sift_gradient_weights)
{
  auto sift = halide::SIFT{};
  const auto& radius = static_cast<std::int32_t>(
      std::round(sift.N / 2 * sift.bin_length_in_scale_unit));
  const auto& diameter = 2 * radius + 1;
  const auto& x0 = -radius;
  const auto& y0 = -radius;

  auto w = Buffer<float>(diameter, diameter);
  w.set_min(x0, y0);

  auto u = Var{"u"};
  auto v = Var{"v"};
  auto w_fn = Func{"gaussian_weights"};
  w_fn(u, v) = sift.gradient_weight(u, v);

  w_fn.realize(w);

  for (auto y = y0; y < y0 + diameter; ++y)
  {
    for (auto x = x0; x < x0 + diameter; ++x)
      std::cout << w(x, y) << " ";
    std::cout << std::endl;
  }
}


BOOST_AUTO_TEST_CASE(test_sift_bin_calculation)
{
  // Create synthetic data.
  const auto& image = make_corner_image();
  const auto& gradients = DO::Sara::gradient(image);

  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  std::transform(gradients.begin(), gradients.end(), mag.begin(),
                 [](const auto& g) { return g.norm(); });
  std::transform(gradients.begin(), gradients.end(), ori.begin(),
                 [](const auto& g) { return std::atan2(g(1), g(0)); });

  // Prepare data for halide.
  const auto& mag_buf = halide::as_buffer(mag);
  const auto& ori_buf = halide::as_buffer(ori);
  const auto& mag_fn = BoundaryConditions::constant_exterior(mag_buf, 0.f);
  const auto& ori_fn = BoundaryConditions::constant_exterior(ori_buf, 0.f);

  // The class we are testing.
  auto sift = halide::SIFT{};

  // Define the functions.
  auto u = Var{"u"};
  auto v = Var{"v"};
  auto k = Var{"k"};

  const Expr x = image.width() / 2.f;
  const Expr y = image.height() / 2.f;
  const Expr s = 1.f;
  const Expr theta = 0.f;


  auto w_fn = Func{"gradient_weights"};
  w_fn(u, v) = sift.gradient_weight(u, v);

  auto patch_fn = Func{"patch"};
  patch_fn(u, v, k) = sift.normalized_gradient_sample(u, v,                  //
                                                      mag_fn, ori_fn, w_fn,  //
                                                      x, y, s, theta);


  const auto& radius = static_cast<int>(
      std::round(sift.bin_length_in_scale_unit * (sift.N + 1) / 2.f));
  const auto& diameter = 2 * radius + 1;
  auto mag_normalized = Buffer<float>(diameter, diameter, 1);
  auto ori_normalized = Buffer<float>(diameter, diameter, 1);
  mag_normalized.set_min(-radius, -radius, 0);
  ori_normalized.set_min(-radius, -radius, 0);
  patch_fn.realize({mag_normalized, ori_normalized});

  auto o = Var{"o"};
  auto j = Var{"j"};
  auto i = Var{"i"};

  auto h_fn = Func{"SIFT"};
  h_fn(o, j, i, k) = sift.compute_bin_value_v3(o, j, i, k, patch_fn);

  const Buffer<float> h = h_fn.realize({8, 4, 4, 1});

  std::cout << std::fixed << std::setw(3) << std::setprecision(2);
  for (auto i = 0; i < 4; ++i)
  {
    for (auto j = 0; j < 4; ++j)
    {
      std::cout << "Processing patch (" << i << ", " << j << ")" << std::endl << std::endl;

      const auto& bin_length_in_scale_unit = sift.bin_length_in_scale_unit;
      const auto& N = sift.N;
      const auto& r = static_cast<int>(bin_length_in_scale_unit);

      const auto& x = std::round((j - N / 2 + 0.5f) * bin_length_in_scale_unit);
      const auto& y = std::round((i - N / 2 + 0.5f) * bin_length_in_scale_unit);

      const auto& xi = static_cast<int>(x);
      const auto& yi = static_cast<int>(y);
      SARA_CHECK(xi);
      SARA_CHECK(yi);
      SARA_CHECK(xi - r);
      SARA_CHECK(yi - r);
      SARA_CHECK(xi + r);
      SARA_CHECK(yi + r);

      std::cout << "mag normalized =" << std::endl;
      for (auto y = yi - r; y <= yi + r; ++y)
      {
        for (auto x = xi - r; x <= xi + r; ++x)
          std::cout << mag_normalized(x, y, 0) << "  ";
        std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << "ori normalized =" << std::endl;
      for (auto y = yi - r; y <= yi + r; ++y)
      {
        for (auto x = xi - r; x <= xi + r; ++x)
          std::cout << ori_normalized(x, y, 0) << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << "SIFT unnormalized =" << std::endl;
      for (auto o = 0; o < 8; ++o)
        std::cout << h(o, j, i, 0) << " ";
      std::cout << std::endl;

      std::cout << std::endl;
    }
  }
}


BOOST_AUTO_TEST_CASE(test_sift_accumulate_subhistogram_v2)
{
  // Create synthetic data.
  const auto& image = make_corner_image();
  const auto& gradients = DO::Sara::gradient(image);

  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  std::transform(gradients.begin(), gradients.end(), mag.begin(),
                 [](const auto& g) { return g.norm(); });
  std::transform(gradients.begin(), gradients.end(), ori.begin(),
                 [](const auto& g) { return std::atan2(g(1), g(0)); });

  // Prepare data for halide.
  const auto& mag_buf = halide::as_buffer(mag);
  const auto& ori_buf = halide::as_buffer(ori);
  const auto& mag_fn = BoundaryConditions::constant_exterior(mag_buf, 0.f);
  const auto& ori_fn = BoundaryConditions::constant_exterior(ori_buf, 0.f);

  // The class we are testing.
  auto sift = halide::SIFT{};

  // Define the functions.
  auto u = Var{"u"};
  auto v = Var{"v"};
  auto k = Var{"k"};

  const Expr x = image.width() / 2.f;
  const Expr y = image.height() / 2.f;
  const Expr s = 1.f;
  const Expr theta = 0.f;


  auto w_fn = Func{"gradient_weights"};
  w_fn(u, v) = sift.gradient_weight(u, v);

  auto wxy_fn = Func{"spatial_weights"};
  wxy_fn(u, v) = sift.spatial_weight(u, v);

  auto patch_fn = Func{"patch"};
  patch_fn(u, v, k) = sift.normalized_gradient_sample(  //
      u, v,                                             //
      mag_fn, ori_fn, w_fn,                             //
      x, y, s, theta                                    //
  );


  const auto& radius = static_cast<int>(
      std::round(sift.bin_length_in_scale_unit * (sift.N + 1) / 2.f));
  const auto& diameter = 2 * radius + 1;
  auto mag_normalized = Buffer<float>(diameter, diameter, 1);
  auto ori_normalized = Buffer<float>(diameter, diameter, 1);
  mag_normalized.set_min(-radius, -radius, 0);
  ori_normalized.set_min(-radius, -radius, 0);
  patch_fn.realize({mag_normalized, ori_normalized});

  auto o = Var{"o"};
  auto ji = Var{"ji"};

  auto h_fn = Func{"SIFT"};
  h_fn(o, ji, k) = 0.f;
  sift.accumulate_subhistogram_v3(h_fn, ji, k, patch_fn, wxy_fn);

  const Buffer<float> h = h_fn.realize({8, 4 * 4, 1});

  std::cout << std::fixed << std::setw(3) << std::setprecision(3);
  for (auto i = 0; i < 4; ++i)
  {
    for (auto j = 0; j < 4; ++j)
    {
      std::cout << "Processing patch (" << i << ", " << j << ")" << std::endl << std::endl;

      const auto& bin_length_in_scale_unit = sift.bin_length_in_scale_unit;
      const auto& N = sift.N;
      const auto& r = static_cast<int>(bin_length_in_scale_unit);

      const auto& x = std::round((j - N / 2 + 0.5f) * bin_length_in_scale_unit);
      const auto& y = std::round((i - N / 2 + 0.5f) * bin_length_in_scale_unit);

      const auto& xi = static_cast<int>(x);
      const auto& yi = static_cast<int>(y);
      SARA_CHECK(xi);
      SARA_CHECK(yi);
      SARA_CHECK(xi - r);
      SARA_CHECK(yi - r);
      SARA_CHECK(xi + r);
      SARA_CHECK(yi + r);

      std::cout << "mag normalized =" << std::endl;
      for (auto y = yi - r; y <= yi + r; ++y)
      {
        for (auto x = xi - r; x <= xi + r; ++x)
          std::cout << mag_normalized(x, y, 0) << "  ";
        std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << "ori normalized =" << std::endl;
      for (auto y = yi - r; y <= yi + r; ++y)
      {
        for (auto x = xi - r; x <= xi + r; ++x)
          std::cout << ori_normalized(x, y, 0) << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;

      std::cout << "SIFT unnormalized =" << std::endl;
      const auto ji = i * N + j;
      for (auto o = 0; o < 8; ++o)
        std::cout << h(o, ji, 0) << " ";
      std::cout << std::endl;

      std::cout << std::endl;
    }
  }
}
