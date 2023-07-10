#pragma once

#include <DO/Sara/FeatureDetectors/SIFT.hpp>

#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>

#include <DO/Sara/RANSAC/RANSACv2.hpp>

#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>


namespace DO::Sara {

  struct RelativePoseEstimator
  {
    const std::array<KeypointList<OERegion, float>, 2>& _keys;
    const std::vector<Match>& _matches;

    int ransac_iterations_max = 1000;
    double ransac_confidence = 0.999;
    double err_thres = 4.;

    // Use Stewenius' algorithm instead of Nister's for now. The polynomial
    // solver must have some convergence problems.
    const RelativePoseSolver<SteweniusFivePointAlgorithm> _solver;
    CheiralAndEpipolarConsistency _inlier_predicate;

    PointCorrespondenceList<double> _X;

    Eigen::Matrix3d _K;
    Eigen::Matrix3d _K_inv;
    Tensor_<bool, 1> _inliers;
    Tensor_<int, 1> _sample_best;
    TwoViewGeometry _geometry;

    RelativePoseEstimator(
        const std::array<KeypointList<OERegion, float>, 2>& keys,
        const std::vector<Match>& matches,
        const v2::BrownConradyDistortionModel<double>& camera)
      : _keys{keys}
      , _matches{matches}
    {
      configure(camera);
    }

    auto configure(const v2::BrownConradyDistortionModel<double>& camera)
        -> void
    {
      _K = camera.calibration_matrix();
      _K_inv = _K.inverse();

      _inlier_predicate.distance.K1_inv = _K_inv;
      _inlier_predicate.distance.K2_inv = _K_inv;
      _inlier_predicate.err_threshold = err_thres;
    }

    auto estimate_relative_pose() -> bool
    {
      print_stage("Estimating the relative pose...");
      if (_matches.empty())
      {
        SARA_DEBUG << "Skipping relative pose estimation\n";
        return false;
      }

      const auto& f0 = features(_keys[0]);
      const auto& f1 = features(_keys[1]);
      const auto u = std::array{
          homogeneous(extract_centers(f0)).cast<double>(),
          homogeneous(extract_centers(f1)).cast<double>()  //
      };
      // List the matches as a 2D-tensor where each row encodes a match 'm' as a
      // pair of point indices (i, j).
      const auto M = to_tensor(_matches);

      _X = PointCorrespondenceList{M, u[0], u[1]};
      auto data_normalizer =
          std::make_optional(Normalizer<TwoViewGeometry>{_K, _K});

      std::tie(_geometry, _inliers, _sample_best) =
          v2::ransac(_X, _solver, _inlier_predicate, ransac_iterations_max,
                     ransac_confidence, data_normalizer, true);
      SARA_DEBUG << "Geometry =\n" << _geometry << std::endl;
      SARA_DEBUG << "inliers count = " << _inliers.flat_array().count()
                 << std::endl;

      return _inliers.flat_array().count() >= 100;
    }
  };

}  // namespace DO::Sara
