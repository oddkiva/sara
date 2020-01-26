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

#ifndef DO_SHAKTI_IMAGEPROCESSING_DIFFERENTIAL_HPP
#define DO_SHAKTI_IMAGEPROCESSING_DIFFERENTIAL_HPP

#include <DO/Shakti/Defines.hpp>

#include <DO/Shakti/MultiArray/MultiArray.hpp>
#include <DO/Shakti/MultiArray/TextureArray.hpp>


namespace DO { namespace Shakti {

  DO_SHAKTI_EXPORT
  MultiArray<Vector2f, 2> gradient(const TextureArray<float>& in);

  DO_SHAKTI_EXPORT
  MultiArray<Vector2f, 2> gradient_polar_coords(const TextureArray<float>& in);

  DO_SHAKTI_EXPORT
  MultiArray<float, 2> gradient_squared_norm(const TextureArray<float>& in);

  DO_SHAKTI_EXPORT
  MultiArray<float, 2> laplacian(const TextureArray<float>& in);

  DO_SHAKTI_EXPORT
  MultiArray<float, 2> squared_norm(const MultiArray<Vector2f, 2>& in);

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  DO_SHAKTI_EXPORT
  void compute_x_derivative(float *out, const float *in, const int *sizes);

  DO_SHAKTI_EXPORT
  void compute_y_derivative(float *out, const float *in, const int *sizes);

  DO_SHAKTI_EXPORT
  void compute_gradient(Vector2f *out, const float *in, const int *sizes);

  DO_SHAKTI_EXPORT
  void compute_gradient_squared_norms(float *out, const float *in,
                                      const int *sizes);

  DO_SHAKTI_EXPORT
  void compute_laplacian(float *out, const float *in, const int *sizes);


} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_IMAGEPROCESSING_DIFFERENTIAL_HPP */
