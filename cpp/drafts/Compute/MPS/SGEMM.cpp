#include "SGEMM.hpp"
#include "SGEMMImpl.hpp"


namespace DO::Sara {

SGEMM::SGEMM()
  : impl{new SGEMMImpl{}}
{
}

void SGEMM::operator()(float alpha, const TensorView_<float, 2>& A,
                       const TensorView_<float, 2>& B, float beta,
                       TensorView_<float, 2>& C) const
{
  impl->operator()(A.rows(), B.cols(), A.cols(), alpha, A.data(), B.data(),
                   beta, C.data());
}

} /* namespace DO::Sara */
