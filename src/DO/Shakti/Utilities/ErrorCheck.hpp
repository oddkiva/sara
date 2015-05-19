#pragma once

#include <stdexcept>

#include <cuda_runtime_api.h>

#include <DO/Sara/Core/StringFormat.hpp>


#define CHECK_CUDA_LAST_RUNTIME_ERROR(msg) \
  do {                                     \
    cudaError __err = cudaGetLastError();  \
    if (__err != cudaSuccess)              \
      throw std::runtime_error(format(     \
        "Fatal error: %s (%s at %s:%d)\n", \
        msg, cudaGetErrorString(__err),    \
        __FILE__, __LINE__));              \
  } while (0)


#define CHECK_CUDA_RUNTIME_ERROR(err) \
  DO::__check_cuda_error(err, __FILE__, __LINE__)


namespace DO {

  inline void __check_cuda_error(cudaError err, const char *file, const int line)
  {
    if (err != cudaSuccess)
      throw std::runtime_error(format(
      "CUDA Runtime API error = %04d from file <%s>, line %i.\n",
      err, file, line).c_str());
  }

} /* namespace DO */