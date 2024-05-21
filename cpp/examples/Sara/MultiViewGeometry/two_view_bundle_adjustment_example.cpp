// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/MultiViewGeometry/BundleAdjustmentProblem.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/Helpers/KeypointMatching.hpp>
#include <DO/Sara/SfM/Helpers/Triangulation.hpp>

#if defined(_WIN32)
#  pragma warning(push, 0)
#endif
#include <ceres/ceres.h>
#if defined(_WIN32)
#  pragma warning(pop)
#endif
#include <ceres/rotation.h>

#include <filesystem>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;

namespace fs = std::filesystem;


struct ReprojectionError
{
  ReprojectionError(double observed_x, double observed_y)
    : observed_x{observed_x}
    , observed_y{observed_y}
  {
  }

  template <typename T>
  bool operator()(const T* const camera,  // (1) camera parameters to optimize.
                  const T* const point,   // (2) 3D points to optimize
                  T* residuals) const
  {
    T p[3];

    auto camera_view = CameraModelView<T>{camera};

    // Rotate the point.
    ceres::AngleAxisRotatePoint(camera_view.angle_axis(), point, p);
    // Translate the point.
    p[0] += camera_view.t()[0];
    p[1] += camera_view.t()[1];
    p[2] += camera_view.t()[2];

    // Normalized camera coordinates.
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    // Apply second and fourth order order radial distortion.
    const auto& l1 = camera_view.l1();
    const auto& l2 = camera_view.l2();
    const auto r2 = xp * xp + yp * yp;
    const auto distortion = T(1) + r2 * (l1 + l2 * r2);
    xp *= distortion;
    yp *= distortion;

    // Apply the internal camera matrix.
    const auto predicted_x = camera_view.fx() * xp + camera_view.x0();
    const auto predicted_y = camera_view.fy() * yp + camera_view.y0();

    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  static ceres::CostFunction* create(const double observed_x,
                                     const double observed_y)
  {
    constexpr auto NumParams = 6 /* camera parameters */ + 3 /* points */;
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, NumParams, 3>{
        new ReprojectionError{observed_x, observed_y}};
  }

  double observed_x;
  double observed_y;
};


GRAPHICS_MAIN()
{
  auto& logger = Logger::get();

  // Load images.
  SARA_LOGI(logger, "Loading images...");
  const auto data_dir = fs::path
  {
#if defined(__APPLE__)
    "/Users/oddkiva/Desktop/Datasets/sfm/castle_int"s
#else
    "/home/david/Desktop/Datasets/sfm/castle_int"s
#endif
  };
  const auto image_ids = std::array<std::string, 2>{"0000", "0001"};
  const auto image_paths = std::array{
      data_dir / (image_ids[0] + ".png"),
      data_dir / (image_ids[1] + ".png")  //
  };
  const auto images = std::array{
      imread<Rgb8>(image_paths[0]),  //
      imread<Rgb8>(image_paths[1])   //
  };

  SARA_LOGI(logger, "Loading the internal camera matrices...");
  const auto K = std::array{
      read_internal_camera_parameters(
          (data_dir / (image_ids[0] + ".png.K")).string())
          .cast<double>(),
      read_internal_camera_parameters(
          (data_dir / (image_ids[1] + ".png.K")).string())
          .cast<double>()  //
  };
  for (auto i = 0; i < 2; ++i)
    SARA_LOGD(logger, "K[{}] =\n{}", i, K[i]);

  SARA_LOGI(logger, "Computing keypoints...");
  const auto image_pyr_params = ImagePyramidParams(-1);
  const auto keypoints = std::array{
      compute_sift_keypoints(images[0].convert<float>(), image_pyr_params),
      compute_sift_keypoints(images[1].convert<float>(), image_pyr_params)  //
  };

  SARA_LOGI(logger, "Matching keypoints...");
  const auto matches = match(keypoints[0], keypoints[1]);

  SARA_LOGI(logger, "Performing data transformations...");
  // Invert the internal camera matrices.
  const auto K_inv = std::array<Eigen::Matrix3d, 2>{
      K[0].inverse(),
      K[1].inverse()  //
  };
  // Tensors of image coordinates.
  const auto& f0 = features(keypoints[0]);
  const auto& f1 = features(keypoints[1]);
  const auto u = std::array{homogeneous(extract_centers(f0)).cast<double>(),
                            homogeneous(extract_centers(f1)).cast<double>()};
  // Tensors of camera coordinates.
  const auto un = std::array{apply_transform(K_inv[0], u[0]),
                             apply_transform(K_inv[1], u[1])};
  static_assert(std::is_same_v<decltype(un[0]), const Tensor_<double, 2>&>);
  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  const auto M = to_tensor(matches);
  const auto X = PointCorrespondenceList{M, un[0], un[1]};

  SARA_LOGI(logger, "Estimating the essential matrix...");
  const auto num_samples = 1000;
  const auto err_thres = 1e-3;
  auto inlier_predicate = InlierPredicate<SampsonEpipolarDistance>{};
  inlier_predicate.err_threshold = err_thres;

  const auto [E, inliers, sample_best] =
      ransac(X, NisterFivePointAlgorithm{}, inlier_predicate, num_samples);

  // Calculate the fundamental matrix.
  SARA_LOGI(logger, "Computing the fundamental matrix...");
  auto F = FundamentalMatrix{};
  F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];


  // Extract the two-view geometry.
  SARA_LOGI(logger, "Estimating the two-view geometry...");
  auto two_view_geometry =
      estimate_two_view_geometry(M, un[0], un[1], E, inliers, sample_best);
  two_view_geometry.C1.K = K[0];
  two_view_geometry.C2.K = K[1];

#if 0
  SARA_LOGI(logger, "Mapping feature GID to match GID...");
  const auto match_index = map_feature_gid_to_match_gid(epipolar_edges);


  SARA_LOGI(logger, "Populating the feature tracks...");
  const auto [feature_graph, components] =
      populate_feature_tracks(views, epipolar_edges);

  // Keep feature tracks of size 2 at least.
  SARA_LOGI(logger, "Checking the feature tracks...");
  const auto feature_tracks =
      filter_feature_tracks(feature_graph, components, views);
  for (const auto& track : feature_tracks)
  {
    std::cout << "Component: " << std::endl;
    std::cout << "Size = " << track.size() << std::endl;
    for (const auto& fgid : track)
    {
      const auto& f = features(views.keypoints[fgid.image_id])[fgid.local_id];
      for (auto [m, m_end] = match_index.equal_range(fgid); m != m_end; ++m)
      {
        const auto point_index = m->second.m;
        std::cout << "- {" << fgid.image_id << ", " << fgid.local_id << "} : "
                  << "STR: " << f.extremum_value << "  "
                  << "TYP: " << int(f.extremum_type) << "  "
                  << "2D:  " << f.center().transpose() << "     "
                  << "3D:  " << two_view_geometry.X.col(point_index).transpose()
                  << std::endl;
      }
    }
    std::cout << std::endl;
  }
  SARA_CHECK(feature_tracks.size());


  // Prepare the bundle adjustment problem formulation .
  auto ba_problem = BundleAdjustmentProblem{};
  ba_problem.populate_data_from_two_view_geometry(
      feature_tracks, views.keypoints, match_index, two_view_geometry);


  // Solve the bundle adjustment problem with Ceres.
  ceres::Problem problem;
  for (int i = 0; i < ba_problem.observations.size(0); ++i)
  {
    auto cost_fn = ReprojectionError::create(ba_problem.observations(i, 0),
                                             ba_problem.observations(i, 1));

    problem.AddResidualBlock(cost_fn, nullptr /* squared loss */,
                             ba_problem.camera_parameters.data() +
                                 ba_problem.camera_indices[i] *
                                     CameraModelView<double>::dof(),
                             ba_problem.points_abs_coords_3d.data() +
                                 ba_problem.point_indices[i] * 3);
  }

  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  auto summary = ceres::Solver::Summary{};
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  SARA_LOGI(logger, "Check the SfM...");
  SARA_DEBUG << "camera_parameters =\n"
             << ba_problem.camera_parameters.matrix() << std::endl;
  SARA_DEBUG << "points =\n"
             << ba_problem.points_abs_coords_3d.matrix() << std::endl;

  // TODO: check the point reprojection errors.
#endif

  return 0;
}
