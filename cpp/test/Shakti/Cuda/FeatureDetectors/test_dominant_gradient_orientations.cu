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

#define BOOST_TEST_MODULE "Halide Backend/Dominant Gradient Orientations"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/DominantOrientations.hpp>
#include <DO/Shakti/Cuda/MultiArray/ManagedMemoryAllocator.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


auto make_corner_image()
{
  auto image = sara::Image<float>{10, 10};
  image.flat_array().fill(0);

  for (auto y = image.height() / 2; y < image.height(); ++y)
    for (auto x = 0; x < image.width() / 2; ++x)
      image(x, y) = 1;

  return image;
}

BOOST_AUTO_TEST_CASE(test_polar_gradients_2d)
{
  auto image = make_corner_image();

  // Image gradients.
  auto mag = sara::Image<float, shakti::ManagedMemoryAllocator>{image.sizes()};
  auto ori = sara::Image<float, shakti::ManagedMemoryAllocator>{image.sizes()};
  // shakti::Cuda::polar_gradient_2d(image, mag, ori);

  std::cout << "image =\n" << image.matrix() << std::endl;
  std::cout << "mag =\n" << mag.matrix() << std::endl;
  std::cout << "ori =\n" << ori.matrix() << std::endl;

  auto grad = sara::gradient(image);
  auto mag2 = grad.cwise_transform([](const auto& v) {  //
    return v.norm();                                    //
  });
  auto ori2 = grad.cwise_transform([](const auto& v) {  //
    return std::atan2(v(1), v(0));                      //
  });

  BOOST_CHECK_SMALL((mag.matrix() - mag2.matrix()).norm(), 1e-6f);
  // Our implementation implements a fast approximate of atan2.
  // With the classical atan2 implementation, the following assert would be
  // true.
  // BOOST_CHECK_SMALL((ori.matrix() - ori2.matrix()).norm(), 1e-6f);
  //
  // In this test case, we satisfy the following order or error magnitude...
  // That's decent.
  BOOST_CHECK_SMALL((ori.matrix() - ori2.matrix()).norm(), 5e-4f);
}

BOOST_AUTO_TEST_CASE(test_box_blur)
{
  constexpr auto O = 36;
  auto h = sara::Image<float>(O, 1);
  h.flat_array().fill(0);
  h(0, 0) = 1;
  h(14, 0) = 1;

  using namespace Halide;

  auto h_buffer = halide::as_buffer(h);
  auto h_fn = BoundaryConditions::constant_exterior(h_buffer, 0);

  const auto o = Var{"o"};
  const auto k = Var{"k"};

  const auto iters = 1;
  auto box_blur_fns = std::vector<Func>(iters);
  for (auto i = 0; i < iters; ++i)
  {
    box_blur_fns[i] = Func{"box_blurred_histogram" + std::to_string(i)};
    if (i == 0)
      box_blur_fns[i](o, k) = halide::box_blur(h_fn, o, k, O);
    else
      box_blur_fns[i](o, k) = halide::box_blur(box_blur_fns[i - 1], o, k, O);
  }

  for (auto i = 0; i < iters - 1; ++i)
    box_blur_fns[i].compute_root();

  Buffer<float> h_blurred = box_blur_fns.back().realize({O, 1});
  BOOST_CHECK_CLOSE(h_blurred(35, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h_blurred(0, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h_blurred(1, 0), 1. / 3., 1e-3f);

  BOOST_CHECK_CLOSE(h_blurred(13, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h_blurred(14, 0), 1. / 3., 1e-3f);
  BOOST_CHECK_CLOSE(h_blurred(15, 0), 1. / 3., 1e-3f);

  for (auto o = 0; o < O; ++o)
    std::cout << h_blurred(o) << " ";
  std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_histogram_of_gradients)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  auto mag_buffer = halide::as_buffer(mag);
  auto ori_buffer = halide::as_buffer(ori);

  using namespace Halide;

  // Prepare the gradient histogram computation.
  //
  // First extend the gradient buffers with zero padding.
  auto mag_fn = BoundaryConditions::constant_exterior(mag_buffer, 0.f);
  auto ori_fn = BoundaryConditions::constant_exterior(ori_buffer, 0.f);

  // Indices of the array buffer.
  Var o{"o"};  // orientation bin index.
  Var k{"k"};  // keypoint index.

  // Number of bins for the orientation histogram.
  constexpr auto O = 36;  // Bins are centered at the following orientations
                          // 0, 10, 20, ..., 350 degrees.
  // Number of keypoints.
  constexpr auto K = 1;

  // Coordinates of the keypoint.
  const Expr x = 5.f;
  const Expr y = 5.f;
  // Image scale (in pixels) at which the keypoint was detected.
  const Expr scale_at_detection = 1.f;

  // To go from this scale to the next discrete scale we multiply the current
  // scale by this geometric factor:
  const Expr scale_residual_max = Halide::pow(2, 1.f / 3.f);  // 1.25992...
  const Expr scale_max = scale_at_detection * scale_residual_max;

  // The refined scale is between this scale and the next scale.
  const Expr scale_residual_exponent = 0.5f;           // Between 0 and 1
  const Expr scale_residual = pow(scale_residual_max,  // Between 1 and ~1.26
                                  scale_residual_exponent);  // ~1.12
  const Expr scale = scale_at_detection * scale_residual;


  // Parameters for the computation of dominant gradient orientations.
  //
  // Orientation bins as a Halide expression.
  const Expr num_orientation_bins = O;
  // Peak ratio threshold.
  constexpr auto peak_ratio_thres = 0.0f;


  // Define the histogram of gradients function.
  auto hog_fn = Func{"gradient_histogram"};
  hog_fn(o, k) = halide::compute_histogram_of_gradients(  //
      mag_fn, ori_fn,                                     //
      x, y, scale,                                        //
      scale_max,                                          //
      o,                                                  //
      num_orientation_bins);                              //

  // Realize the HoG and perform checks.
  {
    Buffer<float> hog = hog_fn.realize({O, K});
    for (auto oi = 0; oi < O; ++oi)
      SARA_DEBUG << "o = " << oi << " hog = " << hog(oi, 0) << std::endl;

    // The image synthesizes a corner at the center, we clearly see
    // three dominant orientations at:
    // * 90 degrees because of horizontal edges.
    // * 180 degrees because of vertical edges.
    // * 135 degrees because of the corner.
    //
    // So we should expect the bins to be zero everywhere except at bins:
    // 9, 13, 14, 18.
    for (auto oi = 0; oi < O; ++oi)
    {
      if (oi == 9 || oi == 18 || oi == 13 || oi == 14)
        BOOST_CHECK(hog(oi, 0) > 0.2);
      else
        BOOST_CHECK(abs(hog(oi, 0)) < 1e-5f);
    }
  }

  // Blur the HoG using iterated box blurs.
  constexpr auto iters = 6;
  auto box_blur_fns = std::vector<Func>(iters);
  for (auto i = 0; i < iters; ++i)
  {
    box_blur_fns[i] = Func{"box_blurred_histogram" + std::to_string(i)};
    auto& prev = i == 0 ? hog_fn : box_blur_fns[i - 1];
    box_blur_fns[i](o, k) = halide::box_blur(prev, o, k, O);
  }
  auto& hog_blurred_fn = box_blur_fns.back();

  // Realize the blurred HoG.
  {
    // Shedule the blur operations.
    hog_fn.compute_root();
    for (auto i = 0; i < iters; ++i)
      box_blur_fns[i].compute_root();

    Buffer<float> hog_blurred = hog_blurred_fn.realize({O, K});
    for (auto oi = 0; oi < O; ++oi)
      SARA_DEBUG << "o = " << oi << " hog_blurred = " << hog_blurred(oi, 0)
                 << std::endl;
  }


  // From now on, calculate the dominant orientations.
  // 1. Localize the peaks.
  // 2. Refine the peaks by calculating the residuals.


  // Localize peaks.
  auto peak_map_fn = Func{"peak_map"};
  peak_map_fn(o, k) =
      halide::is_peak(hog_blurred_fn, o, k, O, peak_ratio_thres);

  // Realize the peak map.
  const Buffer<bool> peak_map_buffer = peak_map_fn.realize({O, K});
  for (auto oi = 0; oi < O; ++oi)
    SARA_DEBUG << "o = " << oi
               << " peak_map_buffer = " << peak_map_buffer(oi, 0) << std::endl;

  // Calculate peak residuals.
  auto peak_residual_fn = Func{"peak_residual"};
  peak_residual_fn(o, k) = halide::compute_peak_residual_map(  //
      hog_blurred_fn,                                          //
      peak_map_fn,                                             //
      o, k, O);

  // Realize the peak residual map.
  const Buffer<float> peak_residual_buffer = peak_residual_fn.realize({O, K});
  for (auto oi = 0; oi < O; ++oi)
    SARA_DEBUG << "o = " << oi
               << " peak_residual_buffer = " << peak_residual_buffer(oi, 0)
               << std::endl;


  // Run the AOT code that runs the equivalent code above.
  {
    auto x_vec = std::vector{5.f};
    auto y_vec = std::vector{5.f};
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
    auto scale_vec = std::vector<float>{scale};

    // Row-major tensors.
    auto peak_map = sara::Tensor_<bool, 2>{1, O};
    auto peak_residuals = sara::Tensor_<float, 2>{1, O};

    DO::Shakti::Halide::dominant_gradient_orientations(
        mag, ori,                                //
        x_vec,                                   //
        y_vec,                                   //
        scale_vec,                               //
        scale_max,                               //
        peak_map,                                //
        peak_residuals,                          //
        /* num_orientation_bins = */ O,          //
        /* gaussian_truncation_factor = */ 3.f,  //
        /* scale_multiplying_factor = */ 1.5f,
        peak_ratio_thres);  //

    for (auto oi = 0; oi < O; ++oi)
    {
      SARA_DEBUG << "o = " << oi << " peak_map = " << peak_map(0, oi)
                 << std::endl;
      BOOST_CHECK(peak_map(0, oi) == peak_map_buffer(oi, 0));
    }

    for (auto oi = 0; oi < O; ++oi)
    {
      SARA_DEBUG << "o = " << oi << " peak_residual = " << peak_residuals(0, oi)
                 << std::endl;
      BOOST_CHECK_CLOSE(peak_residuals(0, oi), peak_residual_buffer(oi, 0),
                        1e-3f);
    }
  }
}

BOOST_AUTO_TEST_CASE(check_halide_impl_with_cpu_impl)
{
  auto image = make_corner_image();

  // Calculate the image gradients in polar coordinates.
  auto mag = sara::Image<float>{image.sizes()};
  auto ori = sara::Image<float>{image.sizes()};
  DO::Shakti::Halide::polar_gradient_2d(image, mag, ori);

  auto polar_grad = sara::Image<Eigen::Vector2f>{image.sizes()};
  std::transform(mag.begin(), mag.end(), ori.begin(), polar_grad.begin(),
                 [](const auto& mag, const auto& ori) {
                   return Eigen::Vector2f{mag, ori};
                 });

  const auto x = 5.f;
  const auto y = 5.f;

  // The scale at which the keypoint is detected.
  const auto scale_at_detection = 1.f;

  // Maximum scale.
  const auto scale_residual_max = std::pow(2.f, 1.f / 3.f);  // 1.25992...
  const auto scale_max = scale_at_detection * scale_residual_max;

  // Keypoint scale.
  const auto scale_residual_exponent = 0.5f;             // Between 0 and 1
  const auto scale = scale_at_detection *                //
                     std::pow(scale_residual_max,        // Between 1 and ~1.26
                              scale_residual_exponent);  // Here: ~1.12

  constexpr auto O = 36;
  constexpr auto num_orientation_bins = O;
  constexpr auto gaussian_truncation_factor = 3.f;
  constexpr auto scale_multiplying_factor = 1.5f;
  constexpr auto peak_ratio_thres = 0.f;

  // Run the original CPU implementation.
  const auto compute_dominant_orientations =
      sara::ComputeDominantOrientations{peak_ratio_thres,            //
                                        gaussian_truncation_factor,  //
                                        scale_multiplying_factor};
  const auto h_cpu = compute_dominant_orientations(polar_grad, x, y, scale);
  SARA_DEBUG << "h_cpu" << std::endl;
  for (const auto& hi : h_cpu)
    std::cout << "orientation = " << hi << " rad" << std::endl;
  std::cout << std::endl;


  // Run the Halide implementation.
  {
    auto x_vec = std::vector{x, x};
    auto y_vec = std::vector{y, y};

    auto scale_vec = std::vector<float>{scale, scale};

    // Row-major tensors.
    auto peak_map = sara::Tensor_<bool, 2>{2, O};
    auto peak_residuals = sara::Tensor_<float, 2>{2, O};

    DO::Shakti::Halide::dominant_gradient_orientations(
        mag, ori,                    //
        x_vec,                       //
        y_vec,                       //
        scale_vec,                   //
        scale_max,                   //
        peak_map,                    //
        peak_residuals,              //
        num_orientation_bins,        //
        gaussian_truncation_factor,  //
        scale_multiplying_factor,    //
        peak_ratio_thres);           //

    SARA_DEBUG << "peak_map =\n" << peak_map.matrix() << std::endl;
    SARA_DEBUG << "peak_residuals =\n" << peak_residuals.matrix() << std::endl;
    BOOST_CHECK_EQUAL(peak_map.matrix().row(0), peak_map.matrix().row(1));
    BOOST_CHECK_EQUAL(peak_residuals.matrix().row(0),
                      peak_residuals.matrix().row(1));

    for (auto o = 0; o < O; ++o)
    {
      if (peak_map(0, o) == 1)
        std::cout << "orientation = "
                  << (o + peak_residuals(0, o)) / O * (2 * M_PI) << std::endl;
    }
  }
}
