#ifndef SGEMM_HPP
#define SGEMM_HPP

#include <DO/Sara/Core/Tensor.hpp>

#include <memory>


namespace DO::Sara {

struct SGEMMImpl;
struct SGEMMImplDeleter
{
  void operator()(const SGEMMImpl*) const;
};

struct SGEMM
{
  SGEMM();

  //! @brief Calculates C such that C = alpha * A * B + beta * C.
  void operator()(float alpha, const TensorView_<float, 2>& A,
                  const TensorView_<float, 2>& B, float beta,
                  TensorView_<float, 2>& C) const;

  std::unique_ptr<SGEMMImpl, SGEMMImplDeleter> impl;
};

} /* namespace DO::Sara */

#endif /* SGEMM_HPP */
