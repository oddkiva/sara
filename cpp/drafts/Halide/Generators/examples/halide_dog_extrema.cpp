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

#include <drafts/Halide/DominantGradientOrientations.hpp>

#include "shakti_halide_gray32f_to_rgb.h"
#include "shakti_halide_rgb_to_gray.h"


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace halide = DO::Shakti::HalideBackend;


auto show_dog_pyramid(sara::ImagePyramid<float>& dog_pyramid)
{
  for (auto o = 0; o < dog_pyramid.num_octaves(); ++o)
  {
    for (auto s = 0; s < dog_pyramid.num_scales_per_octave(); ++s)
    {
      auto& dog = dog_pyramid(s, o);

      auto image_rgb = sara::Image<sara::Rgb8>{dog.sizes()};
      dog.flat_array() = (dog.flat_array() + 1.f) / 2.f;
      auto buffer_gray = halide::as_runtime_buffer<float>(dog);
      auto buffer_rgb = halide::as_interleaved_runtime_buffer(image_rgb);
      shakti_halide_gray32f_to_rgb(buffer_gray, buffer_rgb);

      sara::display(image_rgb);
    }
  }
}

auto show_pyramid(const sara::ImagePyramid<float>& pyramid)
{
  for (auto o = 0; o < pyramid.num_octaves(); ++o)
    for (auto s = 0; s < pyramid.num_scales_per_octave(); ++s)
      sara::display(sara::color_rescale(pyramid(s, o)));
}

auto draw_quantized_extrema(halide::QuantizedExtremaArray& extrema_quantized_so,
                            int i, float oct_scale, float scale)
{
  const auto c0 = extrema_quantized_so.type[i] == 1 ? sara::Blue8 : sara::Red8;
  const auto& x0 = extrema_quantized_so.x[i];
  const auto& y0 = extrema_quantized_so.y[i];
  const auto r0 = scale * std::sqrt(2.f);
  sara::draw_circle(x0 * oct_scale, y0 * oct_scale, r0, c0, 2 + 0);
}

auto draw_dogs(const halide::Pyramid<halide::ExtremaArray>& extrema,
               const halide::Pyramid<halide::DominantOrientationMap>&
                   dominant_orientations_sparse)
{
  for  (const auto& so: extrema.scale_octave_pairs)
  {
    const auto& s = so.first.first;
    const auto& o = so.first.second;

    const auto scale = so.second.first;
    const auto octave_scaling_factor = so.second.second;

    auto eit = extrema.dict.find({s, o});
    if (eit == extrema.dict.end())
      continue;

    const auto& extrema_so = eit->second;
    const auto& ori_so = dominant_orientations_sparse.dict.at({s, o});

    for (auto i = 0u; i < extrema_so.x.size(); ++i)
    {
      const auto c1 = extrema_so.type[i] == 1 ? sara::Cyan8 : sara::Magenta8;

      const auto x1 = extrema_so.x[i] * octave_scaling_factor;
      const auto y1 = extrema_so.y[i] * octave_scaling_factor;
      const auto p1 = Eigen::Vector2f{x1, y1};

      // N.B.: the blob radius is the scale multiplied sqrt(2).
      // http://www.cs.unc.edu/~lazebnik/spring11/lec08_blob.pdf
      const auto r1 = extrema_so.s[i] * octave_scaling_factor * std::sqrt(2.f);

      sara::draw_circle(x1, y1, r1, c1, 2 + 2);

      const auto orientations = ori_so.dominant_orientations(i);
      for (const auto& o: orientations)
      {
        const Eigen::Vector2f p2 = p1 + r1 * Eigen::Vector2f{cos(o), sin(o)};
        sara::draw_line(p1, p2, c1, 2);
      }
    }
  }
}


constexpr auto edge_ratio_thres = 10.f;
constexpr auto extremum_thres = 0.01f;  // 0.01f;
constexpr auto num_orientation_bins = 36;
const auto initial_pyramid_octave = 0;
const auto pyramid_params = sara::ImagePyramidParams(initial_pyramid_octave);


namespace DO::Shakti::HalideBackend {

  struct SIFTExtractor {
    struct Parameters {
      //! @brief Pyramid construction.
      int initial_pyramid_octave = 0;

      //! @brief Extrema detection thresholds.
      float edge_ratio_thres = 10.f;
      float extremum_thres = 0.01f;  // 0.03f;

      //! @brief Dominant gradient orientations.
      int num_orientation_bins = 36;
      float gaussian_truncation_factor = 3.f;
      float scale_multiplying_factor = 1.5f;
      float peak_ratio_thres = 0.0f;
    };

    struct Pipeline {
      Sara::ImagePyramid<float> gaussian_pyramid;
      Sara::ImagePyramid<float> dog_pyramid;
      Sara::ImagePyramid<std::int8_t> dog_extrema_pyramid;
      std::array<Sara::ImagePyramid<float>, 2> gradient_pyramid;

      Pyramid<QuantizedExtremaArray> extrema_quantized;
      Pyramid<ExtremaArray> extrema;

      Pyramid<DominantOrientationDenseMap> dominant_orientations_dense;
      Pyramid<DominantOrientationMap> dominant_orientations;
    };

    Sara::Timer timer;
    Parameters params;
    Pipeline pipeline;

    auto operator()(const Sara::Image<float>& image,
                    const Sara::ImagePyramidParams& pyramid_params)
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

      pipeline.dominant_orientations =
          compress(pipeline.dominant_orientations_dense);
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
  sift_extractor(image, pyramid_params);

  const auto& extrema = sift_extractor.pipeline.extrema;
  const auto& dominant_orientations =
      sift_extractor.pipeline.dominant_orientations;

  const auto num_keypoints = std::accumulate(
      extrema.dict.begin(), extrema.dict.end(), 0,
      [](auto val, const auto& kv) { return val + kv.second.size(); });
  SARA_CHECK(num_keypoints);

  // Show the local extrema.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);
  draw_dogs(extrema, dominant_orientations);
  sara::get_key();
}

auto test_on_video()
{
  using namespace std::string_literals;

#ifdef _WIN32
  const auto video_filepath =
      "C:/Users/David/Desktop/david-archives/gopro-backup-2/GOPR0542.MP4"s;
#elif __APPLE__
  const auto video_filepath = "/Users/david/Desktop/Datasets/sfm/Family.mp4"s;
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
    sift_extractor(frame_gray32f, pyramid_params);
    sara::toc("Oriented DoG");

    sara::tic();
    sara::display(frame);
    draw_dogs(sift_extractor.pipeline.extrema,
              sift_extractor.pipeline.dominant_orientations);
    sara::toc("Display");
  }
}


GRAPHICS_MAIN()
{
  test_on_video();
  return 0;
}
