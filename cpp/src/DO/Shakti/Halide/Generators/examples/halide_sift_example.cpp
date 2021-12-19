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

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Shakti/Halide/Differential.hpp>
#include <DO/Shakti/Halide/LocalExtrema.hpp>
#include <DO/Shakti/Halide/Pyramids.hpp>
#include <DO/Shakti/Halide/RefineExtrema.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>

#include <DO/Shakti/Halide/DominantGradientOrientations.hpp>
#include <DO/Shakti/Halide/Draw.hpp>
#include <DO/Shakti/Halide/Resize.hpp>
#include <DO/Shakti/Halide/SIFT.hpp>

#ifdef USE_SHAKTI_CUDA_VIDEOIO
#  include <DO/Shakti/Cuda/VideoIO.hpp>
#endif

#include "shakti_bgra8u_to_gray32f_cpu.h"
#include "shakti_rgb8u_to_gray32f_cpu.h"


namespace shakti = DO::Shakti;


namespace DO::Shakti::HalideBackend {

  struct SIFTExtractor
  {
    struct Parameters
    {
      //! @brief Pyramid construction.
      int initial_pyramid_octave = 0;

      //! @brief Extrema detection thresholds.
      //! @{
      float edge_ratio_thres = 10.f;
      float extremum_thres = 0.01f;  // 0.03f;
      //! @}

      //! @brief Dominant gradient orientations.
      //! @{
      int num_orientation_bins = 36;
      float gaussian_truncation_factor = 3.f;
      float scale_multiplying_factor = 1.5f;
      float peak_ratio_thres = 0.0f;
      //! @}

      //! @brief SIFT descriptor parameters.
      //! @{
      float bin_length_in_scale_unit = 3.f;
      int N = 4;
      int O = 8;
      //! @}
    };

    struct Pipeline
    {
      Sara::ImagePyramid<float> gaussian_pyramid;
      Sara::ImagePyramid<float> dog_pyramid;
      Sara::ImagePyramid<std::int8_t> dog_extrema_pyramid;
      std::array<Sara::ImagePyramid<float>, 2> gradient_pyramid;

      Pyramid<QuantizedExtremumArray> extrema_quantized;
      Pyramid<ExtremumArray> extrema;

      Pyramid<DominantOrientationDenseMap> dominant_orientations_dense;
      Pyramid<DominantOrientationMap> dominant_orientations;

      // The DoG extremum keypoints.
      Pyramid<OrientedExtremumArray> oriented_extrema;
      // The SIFT descriptors.
      Pyramid<Sara::Tensor_<float, 4>> descriptors;
      Pyramid<Sara::Tensor_<float, 2>> descriptors_v2;
      Pyramid<Sara::Tensor_<float, 3>> descriptors_v3;

      auto num_keypoints() const
      {
        return std::accumulate(
            oriented_extrema.dict.begin(), oriented_extrema.dict.end(), 0ul,
            [](auto val, const auto& kv) { return val + kv.second.size(); });
      }
    };

    Sara::Timer timer;
    Parameters params;
    Pipeline pipeline;

    auto operator()(Sara::ImageView<float>& image)
    {
      timer.restart();
      const auto pyr_params =
          Sara::ImagePyramidParams{params.initial_pyramid_octave};
      pipeline.gaussian_pyramid = gaussian_pyramid(image, pyr_params);
      SARA_DEBUG << "Gaussian pyramid = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      timer.restart();
      pipeline.dog_pyramid = subtract_pyramid(pipeline.gaussian_pyramid);
      SARA_DEBUG << "DoG pyramid = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      timer.restart();
      pipeline.dog_extrema_pyramid = local_scale_space_extrema(  //
          pipeline.dog_pyramid,                                  //
          params.edge_ratio_thres,                               //
          params.extremum_thres);
      SARA_DEBUG << "DoG extrema = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      timer.restart();
      std::tie(pipeline.gradient_pyramid[0],                     //
               pipeline.gradient_pyramid[1]) =                   //
          halide::polar_gradient_2d(pipeline.gaussian_pyramid);  //
      SARA_DEBUG << "Gradient pyramid = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      // Populate the DoG extrema.
      pipeline.extrema_quantized = populate_local_scale_space_extrema(  //
          pipeline.dog_extrema_pyramid);

      // Refine the scale-space localization of each extremum.
      pipeline.extrema = refine_scale_space_extrema(  //
          pipeline.dog_pyramid,                       //
          pipeline.extrema_quantized);                //

      // Estimate the dominant gradient orientations.
      timer.restart();
      dominant_gradient_orientations(pipeline.gradient_pyramid[0],          //
                                     pipeline.gradient_pyramid[1],          //
                                     pipeline.extrema,                      //
                                     pipeline.dominant_orientations_dense,  //
                                     params.num_orientation_bins,           //
                                     params.gaussian_truncation_factor,     //
                                     params.scale_multiplying_factor,       //
                                     params.peak_ratio_thres);
      SARA_DEBUG << "Dominant gradient orientations = " << timer.elapsed_ms()
                 << " ms" << std::endl;

      timer.restart();
      pipeline.dominant_orientations =
          compress(pipeline.dominant_orientations_dense);

      pipeline.oriented_extrema = to_oriented_extremum_array(
          pipeline.extrema, pipeline.dominant_orientations);
      SARA_DEBUG << "Populating oriented extrema = " << timer.elapsed_ms()
                 << " ms" << std::endl;

// #define SIFT_V1
// #define SIFT_V2
// #define SIFT_V3
#define SIFT_V4
#if defined(SIFT_V1)
      SARA_DEBUG << "RUNNING SIFT V1..." << std::endl;
      timer.restart();
      pipeline.descriptors = v1::compute_sift_descriptors(
          pipeline.gradient_pyramid[0], pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema, params.bin_length_in_scale_unit, params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#elif defined(SIFT_V2)
      SARA_DEBUG << "RUNNING SIFT V2..." << std::endl;
      timer.restart();
      pipeline.descriptors_v2 = v2::compute_sift_descriptors(
          pipeline.gradient_pyramid[0], pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema, params.bin_length_in_scale_unit, params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#elif defined(SIFT_V3)
      SARA_DEBUG << "RUNNING SIFT V3..." << std::endl;
      timer.restart();
      pipeline.descriptors_v3 = v3::compute_sift_descriptors(
          pipeline.gradient_pyramid[0], pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema, params.bin_length_in_scale_unit, params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#elif defined(SIFT_V4)
      SARA_DEBUG << "RUNNING SIFT V4..." << std::endl;
      timer.restart();
      pipeline.descriptors_v3 = v4::compute_sift_descriptors(  //
          pipeline.gradient_pyramid[0],                        //
          pipeline.gradient_pyramid[1],                        //
          pipeline.oriented_extrema);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#endif
    }
  };

}  // namespace DO::Shakti::HalideBackend


auto test_on_image()
{
  const auto image_filepath =
#ifdef __APPLE__
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#else
      "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
#endif
  // const auto image_filepath =
  // "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  auto sift_extractor = halide::SIFTExtractor{};
  sift_extractor.params.initial_pyramid_octave = -1;
  auto timer = sara::Timer{};

  timer.restart();
  sift_extractor(image);
  SARA_DEBUG << "Halide SIFT computation time: "  //
             << timer.elapsed_ms() << " ms" << std::endl;
  SARA_CHECK(sift_extractor.pipeline.num_keypoints());

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);
  draw_extrema(sift_extractor.pipeline.oriented_extrema);
  sara::get_key();
}

auto test_on_video(int argc, char** argv)
{
  using namespace std::string_literals;

  const auto video_filepath =
      argc < 2
          ?
#ifdef _WIN32
          "C:/Users/David/Desktop/GOPR0542.MP4"s
#elif __APPLE__
          "/Users/david/Desktop/Datasets/videos/sample10.mp4"s
#else
          "/home/david/Desktop/Datasets/sfm/Family.mp4"s
          // "/home/david/Desktop/GOPR0542.MP4"s;
#endif
          : argv[1];

  // Input and output from Sara.
#ifdef USE_SHAKTI_CUDA_VIDEOIO
  // Initialize CUDA driver.
  DriverApi::init();

  // Create a CUDA context so that we can use the GPU device.
  const auto gpu_id = 0;
  auto cuda_context = DriverApi::CudaContext{gpu_id};
  cuda_context.make_current();

  // nVidia's hardware accelerated video decoder.
  shakti::VideoStream video_stream{video_filepath, cuda_context};
  auto frame =
      sara::Image<sara::Bgra8>{video_stream.width(), video_stream.height()};
  auto device_bgra_buffer =
      DriverApi::DeviceBgraBuffer{video_stream.width(), video_stream.height()};
#else
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
#endif
  auto frame_gray32f = sara::Image<float>{frame.sizes()};

  const auto scale_factor = 1;
  auto frame_downsampled = sara::Image<float>{frame.sizes() / scale_factor};

  // Halide buffers.
#ifdef USE_SHAKTI_CUDA_VIDEOIO
  auto buffer_bgra = halide::as_interleaved_runtime_buffer(frame);
#else
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
#endif
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);

  auto sift_extractor = halide::SIFTExtractor{};

  // Show the local extrema.
  sara::create_window(frame_downsampled.sizes());
  sara::set_antialiasing();

  auto frames_read = 0;
  auto skip = 0;

  while (true)
  {
    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    const auto has_frame = video_stream.read(device_bgra_buffer);
    sara::toc("Read frame");
    if (!has_frame)
      break;

    sara::tic();
    device_bgra_buffer.to_host(frame);
    sara::toc("Copy to host");
#else
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
#endif
    sara::toc("Video Decoding");

    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;

    // Use parallelization and vectorization.
    sara::tic();
#ifdef USE_SHAKTI_CUDA_VIDEOIO
    shakti_bgra8u_to_gray32f_cpu(buffer_bgra, buffer_gray32f);
#else
    shakti_rgb8u_to_gray32f_cpu(buffer_rgb, buffer_gray32f);
#endif
    sara::toc("Grayscale");

    if (scale_factor != 1)
    {
      // Use parallelization and vectorization.
      sara::tic();
      halide::scale(frame_gray32f, frame_downsampled);
      sara::toc("Downsample");
    }

    auto& frame_to_process = scale_factor == 1  //
                                 ? frame_gray32f
                                 : frame_downsampled;

    sara::tic();
    sift_extractor(frame_to_process);
    sara::toc("Oriented DoG");

    sara::tic();
    sara::display(frame_to_process);
    draw_extrema(sift_extractor.pipeline.oriented_extrema);
    sara::toc("Display");
  }
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  // test_on_image();
  test_on_video(argc, argv);
  return 0;
}
