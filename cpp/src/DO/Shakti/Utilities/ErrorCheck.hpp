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

#ifndef DO_SHAKTI_UTILITIES_ERRORCHECK_HPP
#define DO_SHAKTI_UTILITIES_ERRORCHECK_HPP

#include <stdexcept>

#include <cuda_runtime_api.h>

#include <DO/Sara/Core/StringFormat.hpp>


#define SHAKTI_SAFE_CUDA_CALL(err) \
  DO::Shakti::__check_cuda_error(err, __FILE__, __LINE__)


namespace DO { namespace Shakti {

  inline void __check_cuda_error(cudaError err, const char *file, const int line)
  {
    if (err != cudaSuccess)
      throw std::runtime_error(Sara::format(
      "CUDA Runtime API error = %02d from file <%s>, line %i: %s\n",
      err, file, line, cudaGetErrorString(err)).c_str());
  }

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_UTILITIES_ERRORCHECK_HPP */