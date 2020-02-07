// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2015 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_SHAKTI_IMAGEPROCESSING_IMAGEPYRAMID_HPP
#define DO_SHAKTI_IMAGEPROCESSING_IMAGEPYRAMID_HPP

#include <DO/Shakti/MultiArray/Matrix.hpp>


namespace DO { namespace Shakti {

  template <typename T, int N>
  class ImagePyramid
  {
  public:
    using scalar_type = T;

    ImagePyramid(const Vector2i& image_sizes);

  private:
    scalar_type _scale_initial;
    scalar_type _scale_geometric_factor;
    T *_device_data;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_IMAGEPYRAMID_HPP */