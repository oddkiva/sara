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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO::Sara {

  //! @brief Compute SIFT keypoints (DoG+SIFT).
  DO_SARA_EXPORT
  auto compute_sift_keypoints(
      const ImageView<float>& image,
      const ImagePyramidParams& pyramid_params = ImagePyramidParams(),
      float gauss_truncate = 4.f,
      float extremum_thres = 0.01,
      float edge_ratio_thres = 10.f,
      int extremum_refinement_iter = 5,
      bool parallel = false)
      -> KeypointList<OERegion, float>;

} /* namespace DO::Sara */
