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

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>

#ifdef _OMP
#  include <omp.h>
#endif


namespace DO::Sara {

  auto compute_sift_keypoints(const ImageView<float>& image,
                              const ImagePyramidParams& pyramid_params,
                              bool parallel) -> KeypointList<OERegion, float>
  {
    using namespace std;

    // Time everything.
    auto timer = Timer{};
    auto elapsed = 0.;

    // We describe the work flow of the feature detection and description.
    auto DoGs = vector<OERegion>{};
    auto SIFTDescriptors = Tensor_<float, 2>{};

    // 1. Feature extraction.
    timer.restart();
    ComputeDoGExtrema compute_DoGs{pyramid_params, 0.01f};
    auto scale_octave_pairs = vector<Point2i>{};
    DoGs = compute_DoGs(image, &scale_octave_pairs);
    auto dog_detection_time = timer.elapsed_ms();
    elapsed += dog_detection_time;
    SARA_DEBUG << "[DoG        ] " << dog_detection_time << " ms" << std::endl;

    // 2. Feature orientation.
    // Prepare the computation of gradients on gaussians.
    timer.restart();
    auto nabla_G = gradient_polar_coordinates(compute_DoGs.gaussians());
    auto grad_gaussian_time = timer.elapsed_ms();
    elapsed += grad_gaussian_time;
    SARA_DEBUG << "[Gradient   ] " << grad_gaussian_time << " ms" << endl;

    // Find dominant gradient orientations.
    timer.restart();
    ComputeDominantOrientations assign_dominant_orientations;
    assign_dominant_orientations(nabla_G, DoGs, scale_octave_pairs);
    auto ori_assign_time = timer.elapsed_ms();
    elapsed += ori_assign_time;
    SARA_DEBUG << "[Orientation] " << ori_assign_time << " ms" << std::endl;

    if (parallel)
    {
#ifdef _OMP
      const auto max_cpu_threads = omp_get_max_threads();
      SARA_CHECK(max_cpu_threads);
      omp_set_num_threads(max_cpu_threads);
#endif
    }

    // 3. Feature description.
    timer.restart();
    ComputeSIFTDescriptor<> compute_sift;
    SIFTDescriptors = compute_sift(DoGs, scale_octave_pairs, nabla_G, parallel);
    auto sift_description_time = timer.elapsed_ms();

    // 4. Rescale  the feature position and scale $(x, y, \sigma)$ with the
    //    octave scale.
#ifdef _OMP
#  pragma omp parallel for
#endif
    for (size_t i = 0; i != DoGs.size(); ++i)
    {
      const auto octave_scale_factor = static_cast<float>(
          nabla_G.octave_scaling_factor(scale_octave_pairs[i](1)));
      DoGs[i].center() *= octave_scale_factor;
      DoGs[i].shape_matrix /= octave_scale_factor * octave_scale_factor;
    }

    elapsed += sift_description_time;
    SARA_DEBUG << "[Descriptors] " << sift_description_time << " ms" << endl;

    // Summary in terms of computation time.
    SARA_DEBUG << "[SIFT Total] " << elapsed << " ms" << endl;
    SARA_DEBUG << "#{SIFT} = " << SIFTDescriptors.rows() << std::endl;


    return {DoGs, SIFTDescriptors};
  }

} /* namespace DO::Sara */
