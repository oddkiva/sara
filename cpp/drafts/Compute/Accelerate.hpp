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

#pragma once

#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif


namespace DO::Sara {

template <typename T, int O>
inline void gemm(const MultiArrayView<T, 2, O>& A,
                 const MultiArrayView<T, 2, O>& B, MultiArrayView<T, 2, O>& C,
                 T alpha = T(1), T beta = T(0))
{
  constexpr auto order = (O == Eigen::RowMajor) ? CBLAS_ORDER::CblasRowMajor
                                                : CBLAS_ORDER::CblasColMajor;
  const auto m = A.rows();
  const auto n = B.cols();
  const auto k = A.cols();
  const auto transa = CBLAS_TRANSPOSE::CblasNoTrans;
  const auto transb = CBLAS_TRANSPOSE::CblasNoTrans;
  const auto lda = (O == Eigen::RowMajor) ? A.stride(0) : A.stride(1);
  const auto ldb = (O == Eigen::RowMajor) ? B.stride(0) : B.stride(1);
  const auto ldc = (O == Eigen::RowMajor) ? C.stride(0) : C.stride(1);
  if constexpr (std::is_same_v<T, double>)
    ::cblas_dgemm(order, transa, transb, m, n, k, alpha, A.data(), lda,
                  B.data(), ldb, beta, C.data(), ldc);
  else if constexpr (std::is_same_v<T, float>)
    ::cblas_sgemm(order, transa, transb, m, n, k, alpha, A.data(), lda,
                  B.data(), ldb, beta, C.data(), ldc);
}

} /* namespace DO::Sara */
