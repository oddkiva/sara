#define BOOST_TEST_MODULE "Compute/MPS Backend"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <boost/test/unit_test.hpp>

#include "MPS.hpp"


BOOST_AUTO_TEST_CASE(test_mps_gemm)
{
  using namespace DO::Sara;

  for (auto iter = 0; iter < 256; ++iter)
  {
    SARA_DEBUG << "iter = " << iter << std::endl;
    const auto n = (iter + 1) * 8;
    auto r = Tensor_<float, 1>{n * n};
    for (auto i = 0u; i < r.size(); ++i)
      r(i) = float(i);
    // SARA_DEBUG << "r =\n" << r.flat_array().head(8).transpose() << std::endl;

    auto A = Tensor_<float, 2>{n, n};
    A.matrix() = r.reshape(Vector2i{n, n}).matrix();
    // SARA_DEBUG << "A =\n" << A.matrix().topLeftCorner(8, 8) << std::endl;

    auto B = Tensor_<float, 2>{n, n};
    B.flat_array() = r.flat_array() * 1.23456f;
    //SARA_DEBUG << "B =\n" << B.matrix().topLeftCorner(8, 8) << std::endl;

    auto C = Tensor_<float, 2>{n, n};
    auto sgemm = SGEMM{};

    const auto alpha = 1.f;
    const auto beta = 0.f;
    sgemm(alpha, A, B, beta, C);
    // SARA_DEBUG << "C =\n" << C.matrix().topLeftCorner(10, 10) << std::endl;

    auto true_C = Tensor_<float, 2>{n, n};
    true_C.matrix() = A.matrix() * B.matrix();
    // SARA_DEBUG << "true_C =\n" << true_C.matrix().topLeftCorner(10, 10) <<
    // std::endl;

    const double rel_err =
        (true_C.matrix() - C.matrix()).norm() / true_C.matrix().norm();
    BOOST_CHECK_SMALL(rel_err, 1e-6);
  }
}
