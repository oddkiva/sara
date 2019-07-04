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
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


namespace DO::Sara {

auto compute_sift_keypoints(const Image<float>& image)
    -> KeypointList<OERegion, float>
{
  using namespace std;

  // Time everything.
  auto timer = Timer{};
  auto elapsed = 0.;

  // We describe the work flow of the feature detection and description.
  auto DoGs = std::vector<OERegion>{};
  auto SIFTDescriptors = Tensor_<float, 2>{};

  // 1. Feature extraction.
  SARA_DEBUG << "Computing DoG extrema" << endl;
  timer.restart();
  ImagePyramidParams pyr_params(0);
  ComputeDoGExtrema compute_DoGs{pyr_params, 0.01f};
  auto scale_octave_pairs = vector<Point2i>{};
  DoGs = compute_DoGs(image, &scale_octave_pairs);
  auto dog_detection_time = timer.elapsed_ms();
  elapsed += dog_detection_time;
  SARA_DEBUG << "DoG detection time = " << dog_detection_time << " ms" << endl;
  SARA_DEBUG << "DoGs.size() = " << DoGs.size() << endl;

  // 2. Feature orientation.
  // Prepare the computation of gradients on gaussians.
  SARA_DEBUG << "Computing gradients of Gaussians" << endl;
  timer.restart();
  auto nabla_G = gradient_polar_coordinates(compute_DoGs.gaussians());
  auto grad_gaussian_time = timer.elapsed_ms();
  elapsed += grad_gaussian_time;
  SARA_DEBUG << "gradient of Gaussian computation time = " << grad_gaussian_time
             << " ms" << endl;
  SARA_DEBUG << "DoGs.size() = " << DoGs.size() << endl;

  // Find dominant gradient orientations.
  SARA_DEBUG
      << "Assigning (possibly multiple) dominant orientations to DoG extrema"
      << endl;
  timer.restart();
  ComputeDominantOrientations assign_dominant_orientations;
  assign_dominant_orientations(nabla_G, DoGs, scale_octave_pairs);
  auto ori_assign_time = timer.elapsed_ms();
  elapsed += ori_assign_time;
  SARA_DEBUG << "orientation assignment time = " << ori_assign_time << " ms" << endl;
  SARA_DEBUG << "DoGs.size() = " << DoGs.size() << endl;

  // 3. Feature description.
  SARA_DEBUG << "Describe DoG extrema with SIFT descriptors" << endl;
  timer.restart();
  ComputeSIFTDescriptor<> compute_sift;
  SIFTDescriptors = compute_sift(DoGs, scale_octave_pairs, nabla_G);
  auto sift_description_time = timer.elapsed_ms();
  elapsed += sift_description_time;
  SARA_DEBUG << "description time = " << sift_description_time << " ms" << endl;
  SARA_DEBUG << "sifts.size() = " << SIFTDescriptors.size() << endl;

  // Summary in terms of computation time.
  SARA_DEBUG << "Total Detection/Description time" << endl;
  SARA_DEBUG << "SIFT computation time = " << elapsed << " ms" << endl;

  // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
  //    scale.
  for (size_t i = 0; i != DoGs.size(); ++i)
  {
    auto octave_scale_factor =
        nabla_G.octave_scaling_factor(scale_octave_pairs[i](1));
    DoGs[i].center() *= octave_scale_factor;
    DoGs[i].shape_matrix /= pow(octave_scale_factor, 2);
  }

  return {DoGs, SIFTDescriptors};
}

} /* namespace DO::Sara */
