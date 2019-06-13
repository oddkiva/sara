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
  auto decompose(const EssentialMatrix_& E)
  {
    const auto Et = E.transpose();
    const auto ea = Et.row(0);
    const auto eb = Et.row(1);
    const auto ec = Et.col(2);

    const Vector3d eab = ea.cross(eb);
    const Vector3d eac = ea.cross(ec);
    const Vector3d ebc = eb.cross(ec);

    const double mag_eab = eab.norm();
    const double mag_eac = eac.norm();
    const double mag_ebc = ebc.norm();

    Vector3d va, vb, vc;
    if (mag_eab > mag_eac && mag_eab > mag_ebc)
    {
      vc = eab / mag_eab;
      va = ea.normalized();
      vb = vc.cross(va);
    }
    else if (mag_eac > mag_eab && mag_eac > mag_ebc)
    {
      vc = -eac / mag_eab;
      va = ea.normalized();
    }
    else if (mag_ebc > mag_eab && mag_ebc > mag_eab)
    {
    }

    vb = vc.cross(va);

    auto v = std::array<vec_mag_pair_type>{std::make_pair(eab, mag_eab),
                                           std::make_pair(eac, mag_eac),
                                           std::make_pair(ebc, mag_ebc)};
    std::sort(std::begin(v), std::end(v),
              [](const auto& a, const auto& b) { return a.second > b.second; });
  }


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
