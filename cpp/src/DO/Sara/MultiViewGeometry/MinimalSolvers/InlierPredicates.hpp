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
    EpipolarDistance distance;
    double err_threshold;

    CheiralAndEpipolarConsistency() = default;

    CheiralAndEpipolarConsistency(const Model& g)
    {
      set_model(g);
    }

    auto set_model(const Model& g) -> void
    {
      geometry = g;
      distance = EpipolarDistance{essential_matrix(g.C2.R, g.C2.t).matrix()};
    }

    // N.B.: this is not a const method. This triangulates the points from the
    // point correspondences and updates the cheirality.
    template <typename Derived>
    auto operator()(const Eigen::MatrixBase<Derived>& u1,
                    const Eigen::MatrixBase<Derived>& u2)
        -> Eigen::Array<bool, 1, Eigen::Dynamic>
    {
      const auto epipolar_consistent = distance(u1, u2).array() < err_threshold;

      const Matrix34d P1 = geometry.C1;
      const Matrix34d P2 = geometry.C2;

      auto& X = geometry.X;
      auto& cheirality = geometry.cheirality;

      X = triangulate_linear_eigen(P1, P2, u1, u2);
      cheirality = cheirality_predicate(X) &&
                   relative_motion_cheirality_predicate(X, P2);

      return epipolar_consistent && cheirality;
    }

    //! @brief Check the inlier predicate on a list of correspondences.
    template <typename T>
    inline auto operator()(const PointCorrespondenceList<T>& m) const
        -> Array<bool, 1, Dynamic>
    {
      return distance(m._p1.colmajor_view().matrix(),
                      m._p2.colmajor_view().matrix())
                 .array() < err_threshold;
    }
  };

}  // namespace DO::Sara
