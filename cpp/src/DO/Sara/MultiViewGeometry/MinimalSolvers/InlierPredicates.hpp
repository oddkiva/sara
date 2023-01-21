#pragma once

#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp>


namespace DO::Sara {

  //! @brief Joint cheirality and epipolar consistency for RANSAC.
  struct CheiralAndEpipolarConsistency
  {
    using Model = TwoViewGeometry;

    Model geometry;
    SampsonEssentialEpipolarDistance distance;
    double err_threshold;

    CheiralAndEpipolarConsistency() = default;

    CheiralAndEpipolarConsistency(const Model& g)
    {
      set_model(g);
    }

    auto set_model(const Model& g) -> void
    {
      geometry = g;
      distance.set_model(essential_matrix(g.C2.R, g.C2.t).matrix());
    }

    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& u1,
                    const Eigen::MatrixBase<Derived>& u2) const
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      const auto epipolar_consistent = distance(u1, u2).array() < err_threshold;

#if defined(FIXME_NEED_TO_REFACTOR_THE_POINT_CORRESPONDENCE_DATA_STRUCTURE)
      const Matrix34d P1 = geometry.C1.matrix();
      const Matrix34d P2 = geometry.C2.matrix();

      // THE BUG IS HERE: we need to pass backprojected rays and instead we are
      // giving pixel coordinates.
      const auto [X, s1, s2] = triangulate_linear_eigen(P1, P2, u1, u2);
      const auto cheirality = (s1.transpose().array()) > 0 && (s2.transpose().array() > 0);

      return epipolar_consistent && cheirality;
#else
      // But it should still be OK...
      return epipolar_consistent;
#endif
    }

    //! @brief Check the inlier predicate on a list of correspondences.
    template <typename T>
    inline auto operator()(const PointCorrespondenceList<T>& m) const
        -> Array<bool, 1, Dynamic>
    {
      return this->operator()(m._p1.colmajor_view().matrix(),
                              m._p2.colmajor_view().matrix());
    }
  };

}  // namespace DO::Sara
