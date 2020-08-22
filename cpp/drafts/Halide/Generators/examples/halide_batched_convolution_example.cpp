// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <algorithm>
#include <cmath>

#include <omp.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Utilities.hpp>

#include "shakti_convolve_batch_32f.h"
#include "shakti_forward_difference_32f.h"
#include "shakti_halide_rgb_to_gray.h"


namespace sara = DO::Sara;
namespace halide = DO::Shakti::HalideBackend;


struct GaussianKernelBatch
{
  //! @brief Gaussian octave.
  //! @{
  float scale_camera = 1.f;
  float scale_initial = 1.6f;
  float scale_factor = std::pow(2.f, 1 / 3.f);
  int num_scales = 3;
  int gaussian_truncation_factor{4};
  //! @}

  std::vector<float> scales;
  std::vector<float> sigmas;
  sara::Tensor_<float, 2> kernels;
  sara::Tensor_<float, 3> kernels_2d;
  Halide::Runtime::Buffer<float> kernel_x_buffer;
  Halide::Runtime::Buffer<float> kernel_y_buffer;

  auto kernel_size(float sigma) const
  {
    return static_cast<int>(2 * gaussian_truncation_factor * sigma) + 1;
  }

  auto kernel_offset() const
  {
    return -kernels.size(0) / 2;
  }

  auto initialize_kernels()
  {
    scales = std::vector<float>(num_scales + 3);
    for (auto i = 0; i < num_scales + 3; ++i)
      scales[i] = scale_initial * std::pow(scale_factor, i);

    sigmas = std::vector<float>(num_scales + 3);
    sigmas[0] =
        std::sqrt(std::pow(scale_initial, 2) - std::pow(scale_camera, 2));
    for (auto i = 1u; i < sigmas.size(); ++i)
      sigmas[i] = std::sqrt(std::pow(scales[i], 2) - std::pow(scales[0], 2));


    // As separable kernels.
    const auto kernel_size_max = kernel_size(sigmas.back());
    const auto kernel_mid = kernel_size_max / 2;
    kernels.resize(num_scales + 3, kernel_size_max);

    kernels.flat_array().fill(0);
    for (auto n = 0; n < kernels.size(0); ++n)
    {
      const auto& sigma = sigmas[n];
      const auto ksize = kernel_size(sigma);
      const auto kradius = ksize / 2;
      const auto two_sigma_squared = 2 * sigma * sigma;

      for (auto k = 0; k < ksize; ++k)
        kernels(n, k + kernel_mid - kradius) =
            exp(-std::pow(k - kradius, 2) / two_sigma_squared);

      const auto kernel_sum =
          std::accumulate(&kernels(n, kernel_mid - kradius),
                          &kernels(n, kernel_mid - kradius) + ksize, 0.f);

      for (auto k = 0; k < ksize; ++k)
        kernels(n, k + kernel_mid - kradius) /= kernel_sum;
    }

    // Set up on Halide's side.
    kernel_x_buffer = Halide::Runtime::Buffer<float>(
        kernels.data(), kernel_size_max, 1, 1, num_scales + 3);
    kernel_x_buffer.set_min(-kernel_mid, 0, 0, 0);

    kernel_y_buffer = Halide::Runtime::Buffer<float>(
        kernels.data(), 1, kernel_size_max, 1, num_scales + 3);
    kernel_y_buffer.set_min(0, -kernel_mid, 0, 0);
    kernel_x_buffer.set_host_dirty();
    kernel_y_buffer.set_host_dirty();
  }
};


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  auto image = sara::imread<float>(image_filepath);
  auto image_tensor = tensor_view(image).reshape(
      Eigen::Vector4i{1, 1, image.height(), image.width()});
  auto buffer_4d = halide::as_runtime_buffer(image_tensor);

  auto timer = sara::Timer{};
  timer.restart();
  {
    buffer_4d.set_host_dirty();
  }
  const auto elapsed_ms = timer.elapsed_ms();
  SARA_DEBUG << "SIFT octave: " << elapsed_ms << " ms" << std::endl;


  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::tic();
  sara::display(image);
  sara::toc("Display");

  sara::get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto
      video_filepath =  //"/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
      "/Users/david/Desktop/Datasets/videos/sample10.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
  // "/home/david/Desktop/GOPR0542.MP4"s;
#endif


  // ===========================================================================
  // SARA PIPELINE
  //
  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray = sara::Image<float>{frame.sizes()};
  auto frame_gray_tensor =
      tensor_view(frame_gray)
          .reshape(
              Eigen::Vector4i{1, 1, frame_gray.height(), frame_gray.width()});

  auto kernels = GaussianKernelBatch{};
  kernels.initialize_kernels();


  // ===========================================================================
  // HALIDE PIPELINE.
  //
  // RGB-grayscale conversion.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray = halide::as_runtime_buffer(frame_gray);
  auto buffer_gray_4d = halide::as_runtime_buffer(frame_gray_tensor);

  auto x_convolved_batch = Halide::Runtime::Buffer<float>(
      frame.width(), frame.height(), 1, kernels.kernels.size(0));
  auto y_convolved_batch = Halide::Runtime::Buffer<float>(
      frame.width(), frame.height(), 1, kernels.kernels.size(0));

  auto dog_batch = Halide::Runtime::Buffer<float>(
      frame.width(), frame.height(), 1, kernels.kernels.size(0) - 1);

  // Show the local extrema.
  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;

  auto timer = sara::Timer{};
  auto elapsed_ms = double{};

  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    ++frames_read;
    SARA_CHECK(frames_read);

    timer.restart();

    sara::tic();
    shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray);
    sara::toc("CPU rgb to grayscale");

    sara::tic();
    buffer_gray_4d.set_host_dirty();
    sara::toc("Set host dirty");

    sara::tic();
    shakti_convolve_batch_32f(buffer_gray_4d, kernels.kernel_x_buffer,
                              x_convolved_batch);
    sara::toc("Convolving on x-axis");

    sara::tic();
    shakti_convolve_batch_32f(x_convolved_batch, kernels.kernel_y_buffer,
                              y_convolved_batch);
    sara::toc("Convolving on y-axis");

    sara::tic();
    shakti_forward_difference_32f(y_convolved_batch, 3, dog_batch);
    sara::toc("DoG");

// #define CHECK_X_CONV
#ifdef CHECK_X_CONV
    sara::tic();
    x_convolved_batch.copy_to_host();
    sara::toc("Copy conv-x to host");

    for (auto n = 0; n < x_convolved_batch.dim(3).extent(); ++n)
    {
      auto x_slice = x_convolved_batch.sliced(3, n);
      auto x_conv = sara::ImageView<float>(x_slice.data(), frame.sizes());
      sara::display(x_conv);
      sara::draw_string(
          10, 10, sara::format("x-conv: scale[%d] = %f", n, kernels.scales[n]),
          sara::Blue8);
      sara::get_key();
    }
#endif

// #define CHECK_Y_CONV
#ifdef CHECK_Y_CONV
    sara::tic();
    y_convolved_batch.copy_to_host();
    sara::toc("Copy conv-y to host");

    for (auto n = 0; n < y_convolved_batch.dim(3).extent(); ++n)
    {
      auto y_slice = y_convolved_batch.sliced(3, n);
      auto y_conv = sara::ImageView<float>(y_slice.data(), frame.sizes());
      sara::display(y_conv);
      sara::draw_string(10, 10,
                        sara::format("y-conv: scale[%d] = %f "
                                     "sigma[%d] = %f",
                                     n, kernels.scales[n], n,
                                     kernels.sigmas[n]),
                        sara::Blue8);
      sara::get_key();
    }
#endif

// #define CHECK_DOG
#ifdef CHECK_DOG
    sara::tic();
    dog_batch.copy_to_host();
    sara::toc("Copy dog to host");

    sara::tic();
    for (auto n = 0; n < 1 /* dog_batch.dim(3).extent() */; ++n)
    {
      auto dog_slice = dog_batch.sliced(3, n);
      auto dog = sara::ImageView<float>(dog_slice.data(), frame.sizes());
      sara::display(sara::color_rescale(dog));
      sara::draw_string(20, 20,
                        sara::format("dog: scale[%d] = %f "
                                     "sigma[%d] = %f",
                                     n, kernels.scales[n], n,
                                     kernels.sigmas[n]),
                        sara::Blue8);
    }
    sara::toc("Display");
#endif

    sara::display(frame);
  }
}


GRAPHICS_MAIN()
{
  // Optimization.
  omp_set_num_threads(omp_get_max_threads());
  std::ios_base::sync_with_stdio(false);

  // test_on_image();
  test_on_video();
  return 0;
}
