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

#define BOOST_TEST_MODULE "Halide Backend/SIFT descriptor"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/FeatureDescriptors/SIFT.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <DO/Shakti/Halide/Differential.hpp>
#include <DO/Shakti/Halide/SIFT.hpp>


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


auto make_corner_image()
{
  auto image = sara::Image<float>{10, 10};
  image.flat_array().fill(0);

  for (auto y = image.height() / 2; y < image.height(); ++y)
    for (auto x = 0; x < image.width() / 2; ++x)
      image(x, y) = 1;

  return image;
}

BOOST_AUTO_TEST_CASE(test_sift_v1)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  auto polar_gradients = sara::Image<Eigen::Vector2f>{image.sizes()};
  std::transform(mag.begin(), mag.end(), ori.begin(), polar_gradients.begin(),
                 [](const auto& mag, const auto& ori) -> Eigen::Vector2f {
                   return {mag, ori};
                 });

  // SIFT parameters.
  constexpr auto bin_length_in_scale_unit = 3.f;
  constexpr auto N = 4;
  constexpr auto O = 8;

  // Keypoint spatial positions.
  const auto x = image.sizes().x() / 2.f;
  const auto y = image.sizes().y() / 2.f;

  // The scale at which the keypoint is detected.
  auto scale_at_detection = 1.f;

  // Maximum scale.
  auto scale_residual_max = std::pow(2.f, 1.f / 3.f);  // 1.25992...
  auto scale_max = scale_at_detection * scale_residual_max;

  // Keypoint scale.
  auto scale_residual_exponent = 0.5f;             // Between 0 and 1
  auto scale = scale_at_detection *                //
               std::pow(scale_residual_max,        // Between 1 and ~1.26
                        scale_residual_exponent);  // Here: ~1.12

  // Keypoint position.
  const auto theta = 0.f;

  // Row-major tensor.
  //                                         K  I  J  O
  auto descriptor = sara::Tensor_<float, 4>{{1, N, N, O}};

  // Run the AOT code that runs the equivalent code above.
  {
    auto x_vec = std::vector{x};
    auto y_vec = std::vector{y};
    auto scale_vec = std::vector<float>{scale};
    auto theta_vec = std::vector<float>{theta};

    halide::v1::compute_sift_descriptors(mag, ori,                  //
                                         x_vec,                     //
                                         y_vec,                     //
                                         scale_vec,                 //
                                         theta_vec,                 //
                                         scale_max,                 //
                                         descriptor,                //
                                         bin_length_in_scale_unit,  //
                                         N, O);
  }

  auto compute_sift = sara::ComputeSIFTDescriptor<N, O>{};
  const auto descriptor2 = compute_sift(x, y, scale, theta, polar_gradients,
                                        /* do_normalization */ false);

  for (int i = 0; i < descriptor.size(1); ++i)
  {
    for (int j = 0; j < descriptor.size(2); ++j)
    {
      const Eigen::VectorXf h1_ij = descriptor[0][i][j].flat_array().matrix();
      std::cout << "=============================================="
                << std::endl;
      std::cout << "[" << i << "][" << j << "]" << std::endl;
      std::cout << "Halide = " << h1_ij.transpose() << std::endl;

      const Eigen::VectorXf h2_ij = descriptor2.segment(N * O * i + O * j, O);
      std::cout << "Sara   = " << h2_ij.transpose() << std::endl;
      std::cout << std::endl;

      // FIXME.
      // BOOST_CHECK_SMALL((h1_ij - h2_ij).lpNorm<Eigen::Infinity>(), 1e-7f);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_sift_v2)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  auto polar_gradients = sara::Image<Eigen::Vector2f>{image.sizes()};
  std::transform(mag.begin(), mag.end(), ori.begin(), polar_gradients.begin(),
                 [](const auto& mag, const auto& ori) -> Eigen::Vector2f {
                   return {mag, ori};
                 });

  // SIFT parameters.
  constexpr auto bin_length_in_scale_unit = 3.f;
  constexpr auto N = 4;
  constexpr auto O = 8;

  // Keypoint spatial positions.
  const auto x = image.sizes().x() / 2.f;
  const auto y = image.sizes().y() / 2.f;

  // The scale at which the keypoint is detected.
  auto scale_at_detection = 1.f;

  // Maximum scale.
  auto scale_residual_max = std::pow(2.f, 1.f / 3.f);  // 1.25992...
  auto scale_max = scale_at_detection * scale_residual_max;

  // Keypoint scale.
  auto scale_residual_exponent = 0.5f;             // Between 0 and 1
  auto scale = scale_at_detection *                //
               std::pow(scale_residual_max,        // Between 1 and ~1.26
                        scale_residual_exponent);  // Here: ~1.12

  // Keypoint position.
  const auto theta = 0.f;

  // Row-major tensor.
  //                                         K  I JO
  auto descriptor = sara::Tensor_<float, 2>{{1, N * N * O}};

  // Run the AOT code that runs the equivalent code above.
  {
    auto x_vec = std::vector{x};
    auto y_vec = std::vector{y};
    auto scale_vec = std::vector<float>{scale};
    auto theta_vec = std::vector<float>{theta};

    halide::v2::compute_sift_descriptors(mag, ori,                  //
                                         x_vec,                     //
                                         y_vec,                     //
                                         scale_vec,                 //
                                         theta_vec,                 //
                                         scale_max,                 //
                                         descriptor,                //
                                         bin_length_in_scale_unit,  //
                                         N, O);
  }

  auto compute_sift = sara::ComputeSIFTDescriptor<N, O>{};
  const auto descriptor2 =
      compute_sift(x, y, scale, theta, polar_gradients, false);

  // BOOST_CHECK_SMALL((descriptor.flat_array().matrix() - descriptor2).norm(),
  // 1e-2f);

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      const Eigen::VectorXf h1_ij =
          descriptor[0].flat_array().segment(N * O * i + O * j, O);
      std::cout << "=============================================="
                << std::endl;
      std::cout << "[" << i << "][" << j << "]" << std::endl;
      std::cout << "Halide = " << h1_ij.transpose() << std::endl;

      const Eigen::VectorXf h2_ij = descriptor2.segment(N * O * i + O * j, O);
      std::cout << "Sara   = " << h2_ij.transpose() << std::endl;
      std::cout << std::endl;

      BOOST_CHECK_SMALL((h1_ij - h2_ij).lpNorm<Eigen::Infinity>(), 2e-1f);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_sift_v3)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  auto polar_gradients = sara::Image<Eigen::Vector2f>{image.sizes()};
  std::transform(mag.begin(), mag.end(), ori.begin(), polar_gradients.begin(),
                 [](const auto& mag, const auto& ori) -> Eigen::Vector2f {
                   return {mag, ori};
                 });

  // SIFT parameters.
  constexpr auto bin_length_in_scale_unit = 3.f;
  constexpr auto N = 4;
  constexpr auto O = 8;

  // Keypoint spatial positions.
  const auto x = image.sizes().x() / 2.f;
  const auto y = image.sizes().y() / 2.f;

  // The scale at which the keypoint is detected.
  auto scale_at_detection = 1.f;

  // Maximum scale.
  auto scale_residual_max = std::pow(2.f, 1.f / 3.f);  // 1.25992...
  auto scale_max = scale_at_detection * scale_residual_max;

  // Keypoint scale.
  auto scale_residual_exponent = 0.5f;             // Between 0 and 1
  auto scale = scale_at_detection *                //
               std::pow(scale_residual_max,        // Between 1 and ~1.26
                        scale_residual_exponent);  // Here: ~1.12

  // Keypoint position.
  const auto theta = 0.f;

  // Row-major tensor.
  //                                         K  I   J  O
  constexpr auto K = 64;
  auto descriptor = sara::Tensor_<float, 3>{{K, N * N, O}};

  // Run the AOT code that runs the equivalent code above.
  {
    auto x_vec = std::vector(K, x);
    auto y_vec = std::vector(K, y);
    auto scale_vec = std::vector<float>(K, scale);
    auto theta_vec = std::vector<float>(K, theta);

    halide::v3::compute_sift_descriptors(  //
        mag, ori,                          //
        x_vec,                             //
        y_vec,                             //
        scale_vec,                         //
        theta_vec,                         //
        scale_max,                         //
        descriptor,                        //
        bin_length_in_scale_unit,          //
        N, O                               //
    );
  }

  auto compute_sift = sara::ComputeSIFTDescriptor<N, O>{};
  const auto descriptor2 =
      compute_sift(x, y, scale, theta, polar_gradients, false);

  // BOOST_CHECK_SMALL((descriptor.flat_array().matrix() - descriptor2).norm(),
  // 1e-2f);

  for (auto k = 0; k < K; ++k)
  {
    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j)
      {
        const Eigen::VectorXf h1_ij = descriptor[k][i * N + j].flat_array();
        std::cout << "=============================================="
                  << std::endl;
        std::cout << "[" << i << "][" << j << "]" << std::endl;
        std::cout << "Halide = " << h1_ij.transpose() << std::endl;

        const Eigen::VectorXf h2_ij = descriptor2.segment(N * O * i + O * j, O);
        std::cout << "Sara   = " << h2_ij.transpose() << std::endl;
        std::cout << std::endl;

        BOOST_CHECK_SMALL((h1_ij - h2_ij).lpNorm<Eigen::Infinity>(), 1e-6f);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_sift_v4)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  auto polar_gradients = sara::Image<Eigen::Vector2f>{image.sizes()};
  std::transform(mag.begin(), mag.end(), ori.begin(), polar_gradients.begin(),
                 [](const auto& mag, const auto& ori) -> Eigen::Vector2f {
                   return {mag, ori};
                 });

  // SIFT parameters.
  constexpr auto N = 4;
  constexpr auto O = 8;

  // Keypoint spatial positions.
  const auto x = image.sizes().x() / 2.f;
  const auto y = image.sizes().y() / 2.f;

  // The scale at which the keypoint is detected.
  auto scale_at_detection = 1.f;

  // Maximum scale.
  auto scale_residual_max = std::pow(2.f, 1.f / 3.f);  // 1.25992...

  // Keypoint scale.
  auto scale_residual_exponent = 0.5f;             // Between 0 and 1
  auto scale = scale_at_detection *                //
               std::pow(scale_residual_max,        // Between 1 and ~1.26
                        scale_residual_exponent);  // Here: ~1.12

  // Keypoint position.
  const auto theta = 0.f;

  // Row-major tensor.
  //                                         K  I   J  O
  constexpr auto K = 1;
  auto descriptor = sara::Tensor_<float, 3>{{K, N * N, O}};

  // Run the AOT code that runs the equivalent code above.
  {
    auto x_vec = std::vector(K, x);
    auto y_vec = std::vector(K, y);
    auto scale_vec = std::vector<float>(K, scale);
    auto theta_vec = std::vector<float>(K, theta);

    halide::v4::compute_sift_descriptors(  //
        mag, ori,                          //
        x_vec,                             //
        y_vec,                             //
        scale_vec,                         //
        theta_vec,                         //
        descriptor                         //
    );
  }

  auto compute_sift = sara::ComputeSIFTDescriptor<N, O>{};
  const auto descriptor2 =
      compute_sift(x, y, scale, theta, polar_gradients, false);

  for (auto k = 0; k < K; ++k)
  {
    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j)
      {
        const Eigen::VectorXf h1_ij = descriptor[k][i * N + j].vector();
        std::cout << "=============================================="
                  << std::endl;
        std::cout << "[" << k << "][" << i << "][" << j << "]" << std::endl;
        std::cout << "Halide = " << h1_ij.transpose() << std::endl;

        const Eigen::VectorXf h2_ij = descriptor2.segment(N * O * i + O * j, O);
        std::cout << "Sara   = " << h2_ij.transpose() << std::endl;
        std::cout << std::endl;

        // TODO: understand better why...
        // BOOST_CHECK_SMALL((h1_ij - h2_ij).lpNorm<Eigen::Infinity>(), 1e-6f);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(test_sift_v5)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  // SIFT parameters.
  static constexpr auto N = 4;
  static constexpr auto O = 8;

  // Keypoint spatial positions.
  const auto x = image.sizes().x() / 2.f;
  const auto y = image.sizes().y() / 2.f;

  // The scale at which the keypoint is detected.
  static const auto scale_at_detection = 1.f;

  // Maximum scale.
  auto scale_residual_max = std::pow(2.f, 1.f / 3.f);  // 1.25992...

  // Keypoint scale.
  auto scale_residual_exponent = 0.5f;             // Between 0 and 1
  auto scale = scale_at_detection *                //
               std::pow(scale_residual_max,        // Between 1 and ~1.26
                        scale_residual_exponent);  // Here: ~1.12

  // Keypoint position.
  const auto theta = 0.f;

  // Row-major tensor.
  //                                         K  I   J  O
  static constexpr auto K = 1;

  auto descriptor_v4 = sara::Tensor_<float, 3>{{K, N * N, O}};
  auto descriptor_v5 = sara::Tensor_<float, 3>{{K, N * N, O}};
  descriptor_v4.flat_array().setZero();
  descriptor_v5.flat_array().setZero();

  // Run SIFT V4.
  {
    auto x_vec = std::vector(K, x);
    auto y_vec = std::vector(K, y);
    auto scale_vec = std::vector<float>(K, scale);
    auto theta_vec = std::vector<float>(K, theta);
    halide::v4::compute_sift_descriptors(  //
        mag, ori,                          //
        x_vec,                             //
        y_vec,                             //
        scale_vec,                         //
        theta_vec,                         //
        descriptor_v4                      //
    );
  }

  // Run SIFT V5.
  {
    auto x_vec = std::vector(K, x);
    auto y_vec = std::vector(K, y);
    auto scale_vec = std::vector<float>(K, scale);
    auto theta_vec = std::vector<float>(K, theta);
    halide::v5::compute_sift_descriptors(  //
        mag, ori,                          //
        x_vec,                             //
        y_vec,                             //
        scale_vec,                         //
        theta_vec,                         //
        descriptor_v5                      //
    );
  }

  BOOST_CHECK_SMALL((descriptor_v4.row_vector() - descriptor_v5.row_vector()).norm(), 1e-6f);
}
