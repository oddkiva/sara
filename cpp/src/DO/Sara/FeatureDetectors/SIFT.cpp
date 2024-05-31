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

#include <DO/Sara/FeatureDetectors/SIFT.hpp>

#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/FeatureDescriptors/Orientation.hpp>
#include <DO/Sara/FeatureDescriptors/SIFT.hpp>
#include <DO/Sara/FeatureDetectors/DoG.hpp>
#include <DO/Sara/Logging/Logger.hpp>

#ifdef _OPENMP
#  include <omp.h>
#endif


namespace DO::Sara {

  auto compute_sift_keypoints(const ImageView<float>& image,
                              const ImagePyramidParams& pyramid_params,
                              float gauss_truncate, float extremum_thres,
                              float edge_ratio_thres,
                              int extremum_refinement_iter, bool parallel)
      -> KeypointList<OERegion, float>
  {
    // Time everything.
    auto& logger = Logger::get();
    auto timer = Timer{};
    auto elapsed = 0.;

    // We describe the work flow of the feature detection and description.
    auto DoGs = std::vector<OERegion>{};
    auto SIFTDescriptors = Tensor_<float, 2>{};

    // 1. Feature extraction.
    timer.restart();
    auto compute_DoGs = ComputeDoGExtrema{
        pyramid_params,           //
        gauss_truncate,           //
        extremum_thres,           //
        edge_ratio_thres,         //
        extremum_refinement_iter  //
    };
    auto scale_octave_pairs = std::vector<Point2i>{};
    DoGs = compute_DoGs(image, &scale_octave_pairs);
    const auto dog_detection_time = timer.elapsed_ms();
    elapsed += dog_detection_time;
    SARA_LOGD(logger, "[DoG        ] {:0.2f} ms", dog_detection_time);

    // 2. Feature orientation.
    // Prepare the computation of gradients on gaussians.
    timer.restart();
    auto nabla_G = gradient_polar_coordinates(compute_DoGs.gaussians());
    const auto grad_gaussian_time = timer.elapsed_ms();
    elapsed += grad_gaussian_time;
    SARA_LOGD(logger, "[Gradient   ] {:0.2f} ms", grad_gaussian_time);

    // Find dominant gradient orientations.
    timer.restart();
    ComputeDominantOrientations assign_dominant_orientations;
    assign_dominant_orientations(nabla_G, DoGs, scale_octave_pairs);
    const auto ori_assign_time = timer.elapsed_ms();
    elapsed += ori_assign_time;
    SARA_LOGD(logger, "[Orientation] {:0.2f} ms", ori_assign_time);

    if (parallel)
    {
#ifdef _OPENMP
      const auto max_cpu_threads = omp_get_max_threads();
      SARA_CHECK(max_cpu_threads);
      omp_set_num_threads(max_cpu_threads);
#endif
    }

    // 3. Feature description.
    timer.restart();
    ComputeSIFTDescriptor<> compute_sift;
    SIFTDescriptors = compute_sift(DoGs, scale_octave_pairs, nabla_G, parallel);
    const auto sift_description_time = timer.elapsed_ms();

    // 4. Rescale  the feature position and scale $(x, y, \sigma)$ with the
    //    octave scale.
    SARA_LOGD(logger, "[Descriptors] {} keypoints to describe with SIFT", DoGs.size());
    for (size_t i = 0; i != DoGs.size(); ++i)
    {
      const auto octave_scale_factor = static_cast<float>(
          nabla_G.octave_scaling_factor(scale_octave_pairs[i](1)));
      DoGs[i].center() *= octave_scale_factor;
      DoGs[i].shape_matrix /= octave_scale_factor * octave_scale_factor;
    }

    elapsed += sift_description_time;
    SARA_LOGD(logger, "[Descriptors] {:0.2f} ms", sift_description_time);

    // Summary in terms of computation time.
    SARA_LOGD(logger, "[SIFT Total] {:0.2f} ms", elapsed);
    SARA_LOGD(logger, "[SIFT Total] {} features", SIFTDescriptors.rows());

    return {DoGs, SIFTDescriptors};
  }

} /* namespace DO::Sara */
