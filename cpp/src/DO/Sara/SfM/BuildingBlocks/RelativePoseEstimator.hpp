#pragma once

#include <DO/Sara/Features/KeypointList.hpp>

#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>

#include <DO/Sara/RANSAC/RANSACv2.hpp>


namespace DO::Sara {

  struct RelativePoseEstimator
  {
    int ransac_iterations_max = 1000;
    double ransac_confidence = 0.999;
    double err_thres = 4.;

    // Use Stewenius' algorithm instead of Nister's for now. The polynomial
    // solver must have some convergence problems.
    const RelativePoseSolver<SteweniusFivePointAlgorithm> _solver;
    CheiralAndEpipolarConsistency _inlier_predicate;

    Eigen::Matrix3d _K;
    Eigen::Matrix3d _K_inv;

    auto configure(const v2::BrownConradyDistortionModel<double>& camera)
        -> void;

    auto estimate_relative_pose(const KeypointList<OERegion, float>& src_keys,
                                const KeypointList<OERegion, float>& dst_keys,
                                std::vector<Match>& matches) const
        -> std::tuple<TwoViewGeometry, Tensor_<bool, 1>, Tensor_<int, 1>>;
  };

}  // namespace DO::Sara
