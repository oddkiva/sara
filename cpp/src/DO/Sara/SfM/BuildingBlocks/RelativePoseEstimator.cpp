#include <DO/Sara/SfM/BuildingBlocks/RelativePoseEstimator.hpp>

#include <DO/Sara/Logging/Logger.hpp>

#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>

using namespace DO::Sara::v2;

auto RelativePoseEstimator::configure(
    const v2::BrownConradyDistortionModel<double>& camera) -> void
{
  _K = camera.calibration_matrix();
  _K_inv = _K.inverse();

  _inlier_predicate.distance.K1_inv = _K_inv;
  _inlier_predicate.distance.K2_inv = _K_inv;
  _inlier_predicate.err_threshold = err_thres;
}

auto RelativePoseEstimator::estimate_relative_pose(
    const KeypointList<OERegion, float>& src_keys,
    const KeypointList<OERegion, float>& dst_keys,
    std::vector<Match>& matches) const
    -> std::tuple<TwoViewGeometry, Tensor_<bool, 1>, Tensor_<int, 1>>
{
  auto logger = Logger::get();

  SARA_LOGD(logger, "Estimating the relative pose...");
  if (matches.empty())
  {
    SARA_LOGD(logger, "Skipping relative pose estimation");
    return {};
  }

  const auto& f0 = features(src_keys);
  const auto& f1 = features(dst_keys);
  const auto u = std::array{
      homogeneous(extract_centers(f0)).cast<double>(),
      homogeneous(extract_centers(f1)).cast<double>()  //
  };
  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  const auto M = to_tensor(matches);

  const auto X = PointCorrespondenceList{M, u[0], u[1]};
  auto data_normalizer =
      std::make_optional(Normalizer<TwoViewGeometry>{_K, _K});

  return v2::ransac(X, _solver, _inlier_predicate, ransac_iterations_max,
                    ransac_confidence, data_normalizer, true);
}
