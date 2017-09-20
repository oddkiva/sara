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

    EssentialMatrix() = default;

    friend auto extract_candidate_camera_motions(const EssentialMatrix& E)
        -> std::pair<std::array<matrix_type, 2>, std::array<point_type, 2>>
    {
      auto svd =
          JacobiSVD<matrix_type>{E, Eigen::ComputeFullU | Eigen::ComputeFullV};

      const auto U = svd.matrixU();
      const auto W = svd.singularValues().diag();
      const auto V = svd.matrixU();
      const auto t = svd.matrixU().col(2);
      return std::make_pair(
          std::array<matrix_type, 2>{U * W * V.transpose(),
                                     U * W.transpose() * V.transpose()},
          std::array<point_type, 2>{t, -t});
    }
  };

} /* namespace Sara */
} /* namespace DO */
