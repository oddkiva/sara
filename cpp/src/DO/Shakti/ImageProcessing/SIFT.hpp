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

#ifndef DO_SHAKTI_IMAGEPROCESSING_SIFT_HPP
#define DO_SHAKTI_IMAGEPROCESSING_SIFT_HPP

#include <DO/Shakti/Defines.hpp>

#include <DO/Shakti/MultiArray.hpp>


namespace DO { namespace Shakti {

  class DO_SHAKTI_EXPORT DenseSiftComputer
  {
  public:
    DenseSiftComputer();

    MultiArray<Vector<float, 128>, 2>
    operator()(const TextureArray<Vector2f>& gradients) const;

    void operator()(float* out, const float* in, const int* sizes) const;

  private:
    float _bin_scale_unit_length = 3.f;
    //! @brief Maximum value for a descriptor bin value to remain robust w.r.t.
    //! illumination changes.
    float _max_bin_value = 0.2f;
    float _sigma = 1.6f;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_SIFT_HPP */
