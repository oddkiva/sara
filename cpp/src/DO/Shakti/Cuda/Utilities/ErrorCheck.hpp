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

#pragma once

#include <DO/Shakti/Cuda/Utilities/StringFormat.hpp>

#include <termcolor/termcolor.hpp>

#include <cuda_runtime_api.h>

#include <iostream>
#include <stdexcept>


#define SHAKTI_SAFE_CUDA_CALL(err)                                             \
  DO::Shakti::__check_cuda_error(err, __FILE__, __LINE__)

#define SHAKTI_CHECK_LAST_ERROR()                                              \
  DO::Shakti::__check_cuda_error(cudaPeekAtLastError(), __FILE__, __LINE__)

#define SHAKTI_SYNCHRONIZE()                                                   \
  DO::Shakti::__check_cuda_error(cudaDeviceSynchronize(), __FILE__, __LINE__)

#define SHAKTI_SYNCHRONIZED_CHECK()                                            \
  SHAKTI_CHECK_LAST_ERROR();                                                   \
  SHAKTI_SYNCHRONIZE()

#define SHAKTI_STDOUT                                                          \
  std::cout << termcolor::bold << termcolor::green << "[" << __FUNCTION__      \
            << ":" << __LINE__ << "] " << termcolor::reset

#define SHAKTI_STDERR                                                          \
  std::cerr << termcolor::bold << termcolor::red << "[" << __FUNCTION__ << ":" \
            << __LINE__ << "] " << termcolor::reset


namespace DO { namespace Shakti {

  inline void __check_cuda_error(cudaError err, const char* file,
                                 const int line)
  {
    if (err != cudaSuccess)
    {
      throw std::runtime_error{
          Shakti::format("[SYNCHRONIZATION] CUDA Runtime API error = %02d from "
                         "file <%s>, line %i: %s\n",
                         err, file, line, cudaGetErrorString(err))};

#ifdef _WIN32
      std::cerr
          << Shakti::format(
                 "CUDA Runtime API error = %02d from file <%s>, line %i: %s\n",
                 err, file, line, cudaGetErrorString(err))
          << std::endl;
      std::abort();
#endif
    }

#ifdef SYNCHRONIZE_CUDA_DEVICE
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
      std::cerr << Shakti::format(
                       "[SYNCHRONIZATION] CUDA Runtime API error = %02d from "
                       "file <%s>, line %i: %s\n",
                       err, file, line, cudaGetErrorString(err))
                << std::endl;
      std::abort();
    }
#endif
  }

}}  // namespace DO::Shakti
