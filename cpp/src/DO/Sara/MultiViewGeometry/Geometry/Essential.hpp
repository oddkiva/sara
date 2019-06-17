#pragma once

#include <DO/Sara/MultiViewGeometry/Geometry/Fundamental.hpp>


namespace DO { namespace Sara {

  template <typename T = double>
  class EssentialMatrix : public FundamentalMatrix<T>
  {
    using base_type = FundamentalMatrix<T>;

  public:
    using matrix_type = typename base_type::matrix_type;
    using point_type = typename base_type::point_type;
    using vector_type = point_type;

    using motion_type = std::pair<matrix_type, vector_type>;

    EssentialMatrix() = default;
  };

  template <typename EssentialMatrix_>
  auto extract_candidate_camera_motions(const EssentialMatrix_& E)
      -> std::array<typename EssentialMatrix_::motion_type, 4>
  {
    using matrix_type = typename EssentialMatrix_::matrix_type;
    using point_type = typename EssentialMatrix_::point_type;

    auto svd =
        JacobiSVD<matrix_type>{E, Eigen::ComputeFullU | Eigen::ComputeFullV};

    const auto U = svd.matrixU();
    const auto W = svd.singularValues().diag();
    const auto V = svd.matrixU();
    const auto t = svd.matrixU().col(2);

    const auto R1 = U * W * V.transpose();
    const auto R2 = U * W.transpose() * V.transpose();
    const auto t1 = t;
    const auto t2 = -t;

    return {{R1, t1}, {R2, t1}, {R1, t2}, {R2, t2}};
  }

} /* namespace Sara */
} /* namespace DO */
