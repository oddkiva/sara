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

#include <Eigen/Sparse>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Halide/Differential.hpp>
#include <drafts/Halide/LocalExtrema.hpp>
#include <drafts/Halide/Pyramids.hpp>
#include <drafts/Halide/RefineExtrema.hpp>
#include <drafts/Halide/Utilities.hpp>

#include <drafts/Halide/Draw.hpp>
#include <drafts/Halide/DominantGradientOrientations.hpp>
#include <drafts/Halide/SIFT.hpp>


namespace shakti = DO::Shakti;


namespace DO::Shakti::HalideBackend {

  struct SIFTExtractor {
    struct Parameters {
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

    struct Pipeline {
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
    };

    Sara::Timer timer;
    Parameters params;
    Pipeline pipeline;

    auto operator()(const Sara::Image<float>& image)
    {
      timer.restart();
      const auto pyr_params =
          Sara::ImagePyramidParams{params.initial_pyramid_octave};
      pipeline.gaussian_pyramid = gaussian_pyramid(image, pyr_params);
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

      timer.restart();
      pipeline.descriptors = compute_sift_descriptors(
          pipeline.gradient_pyramid[0],
          pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema,
          params.bin_length_in_scale_unit,
          params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms()
                 << " ms" << std::endl;
    }
  };

}


auto test_on_image()
{
  const auto image_filepath =
      "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg";
  // const auto image_filepath =
  // "/Users/david/GitLab/DO-CV/sara/cpp/drafts/MatchPropagation/cpp/examples/shelves/shelf-1.jpg";
  auto image = sara::imread<float>(image_filepath);

  auto sift_extractor = halide::SIFTExtractor{};
  sift_extractor(image);

  const auto& extrema = sift_extractor.pipeline.extrema;
  const auto num_keypoints = std::accumulate(
      extrema.dict.begin(), extrema.dict.end(), 0,
      [](auto val, const auto& kv) { return val + kv.second.size(); });
  SARA_CHECK(num_keypoints);

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);
  draw_extrema(sift_extractor.pipeline.oriented_extrema);
  sara::get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  // const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
  const auto video_filepath = "/Users/david/Desktop/Datasets/humanising-autonomy/field.mp4"s;
#else
  const auto video_filepath = "/home/david/Desktop/Datasets/sfm/Family.mp4"s;
#endif

  // Input and output from Sara.
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_gray32f = sara::Image<float>{video_stream.sizes()};

  // Halide buffers.
  auto buffer_rgb = halide::as_interleaved_runtime_buffer(frame);
  auto buffer_gray32f = halide::as_runtime_buffer<float>(frame_gray32f);

  auto sift_extractor = halide::SIFTExtractor{};

  // Show the local extrema.
  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  while (true)
  {
    sara::tic();
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    sara::toc("Video Decoding");

    // Use parallelization and vectorization.
    sara::tic();
    shakti_halide_rgb_to_gray(buffer_rgb, buffer_gray32f);
    sara::toc("Grayscale");

    sara::tic();
    sift_extractor(frame_gray32f);
    sara::toc("Oriented DoG");

    sara::tic();
    sara::display(frame);
    draw_extrema(sift_extractor.pipeline.oriented_extrema);
    sara::toc("Display");
  }
}


GRAPHICS_MAIN()
{
  test_on_video();
  return 0;
}
