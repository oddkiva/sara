#pragma once

#include "MultiArray/Matrix.hpp"
#include "MultiArray/Offset.hpp"


//! Question: understand whether there is a performance penalty when adding
//! more parameters to the CUDA kernel function.
/*!
    Presumably the '__constant__' keyword hints at this problem.
 */

//! Avoid warp divergence.
//! The key idea is to partition the data when conditional statement appears.
//! However, profiling is needed in order to find out whether a specific part of
//! CUDA code is worth optimizing.

template <typename T, int N>
__global__
void init(T *data, int size)
{
  int i = DO::Shakti::offset<N>();
  data[i] = 1;
}

template <typename T, int N>
__global__
void gradient(const T *src, T *dst, int size, DO::Shakti::Vector<int, N> strides)
{
  int i = DO::Shakti::offset<N>();

  DO::Shakti::Vector<T, N> grad_f;
#pragma unroll
  for (int i = 0; i < N; ++i)
  {
    int ip = i + strides(i);
    int im = i - strides(i);
    if (ip >= size)
      ip = i;
    if (im < 0)
      im = i;
    grad_f(i) = (src[ip] - src[im]) / 2;
  }

  dst[i] = sqrt(grad_f.dot(grad_f));
}

template <typename T, int N>
__global__
void laplacian(const T *src, T *dst, int size, DO::Shakti::Vector<int, N> strides)
{
  int i = DO::Shakti::offset<N>();

  dst[i] = -2*N*src[i];
#pragma unroll
  for (int i = 0; i < N; ++i)
  {
    int ip = i + strides(i);
    int im = i - strides(i);
    if (ip >= size)
      ip = i;
    if (im < 0)
      im = i;
    dst[i] = src[ip] + src[im];
  }
}

//template <typename T, int N>
//__global__
//void hessian(const T *src, Matrix<T, N, N> *dst, int size,
//             Shakti::Vector<int, N> strides)
//{
//  int off = offset<N>();
//
//  for (int i = 0; i < N; ++i)
//  {
//    for (int j = i; j < N; ++j)
//    {
//      dst[off](i,j) = ....
//    }
//  }
//}