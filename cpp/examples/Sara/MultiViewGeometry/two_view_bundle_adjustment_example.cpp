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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/MultiViewGeometry/BundleAdjustmentProblem.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>

#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


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


auto map_feature_gid_to_match_gid(const EpipolarEdgeAttributes& epipolar_edges)
{
  auto mapping = std::multimap<FeatureGID, MatchGID>{};
  for (const auto& ij : epipolar_edges.edge_ids)
  {
    const auto view_i = epipolar_edges.edges[ij].first;
    const auto view_j = epipolar_edges.edges[ij].second;
    const auto& M_ij = epipolar_edges.matches[ij];
    const auto& E_inliers_ij = epipolar_edges.E_inliers[ij];
    const auto& two_view_geometry_ij = epipolar_edges.two_view_geometries[ij];

    for (auto m = 0; m < int(M_ij.size()); ++m)
    {
      if (E_inliers_ij(m) && two_view_geometry_ij.cheirality(m))
      {
        mapping.insert({{view_i, M_ij[m].x_index()}, {ij, m}});
        mapping.insert({{view_j, M_ij[m].y_index()}, {ij, m}});
      }
    }
  }

  return mapping;
}


GRAPHICS_MAIN()
{
  // Use the following data structure to load images, keypoints, camera
  // parameters.
  auto views = ViewAttributes{};

  // Load images.
  print_stage("Loading images...");
  const auto data_dir =
#ifdef __APPLE__
      "/Users/david/Desktop/Datasets/sfm/castle_int"s;
#else
      "/home/david/Desktop/Datasets/sfm/castle_int"s;
#endif
  views.image_paths = {
      data_dir + "/" + "0000.png",
      data_dir + "/" + "0001.png",
  };
  views.read_images();


  print_stage("Loading the internal camera matrices...");
  views.cameras.resize(2 /* views */);
  views.cameras[0].K =
      read_internal_camera_parameters(data_dir + "/" + "0000.png.K")
          .cast<double>();
  views.cameras[1].K =
      read_internal_camera_parameters(data_dir + "/" + "0001.png.K")
          .cast<double>();


  print_stage("Computing keypoints...");
  views.keypoints = {compute_sift_keypoints(views.images[0].convert<float>()),
                     compute_sift_keypoints(views.images[1].convert<float>())};

  // Use the following data structures to store the epipolar geometry data.
  auto epipolar_edges = EpipolarEdgeAttributes{};
  epipolar_edges.initialize_edges(2 /* views */);
  epipolar_edges.resize_fundamental_edge_list();
  epipolar_edges.resize_essential_edge_list();


  print_stage("Matching keypoints...");
  epipolar_edges.matches = {match(views.keypoints[0], views.keypoints[1])};
  const auto& matches = epipolar_edges.matches[0];


  print_stage("Performing data transformations...");
  // Invert the internal camera matrices.
  const auto K_inv = std::array<Matrix3d, 2>{views.cameras[0].K.inverse(),
                                             views.cameras[1].K.inverse()};
  // Tensors of image coordinates.
  const auto& f0 = features(views.keypoints[0]);
  const auto& f1 = features(views.keypoints[1]);
  const auto u = std::array{homogeneous(extract_centers(f0)).cast<double>(),
                            homogeneous(extract_centers(f1)).cast<double>()};
  // Tensors of camera coordinates.
  const auto un = std::array{apply_transform(K_inv[0], u[0]),
                             apply_transform(K_inv[1], u[1])};
  static_assert(std::is_same_v<decltype(un[0]), const Tensor_<double, 2>&>);
  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  const auto M = to_tensor(matches);


  print_stage("Estimating the essential matrix...");
  auto& E = epipolar_edges.E[0];
  auto& num_samples = epipolar_edges.E_num_samples[0];
  auto& err_thres = epipolar_edges.E_noise[0];
  auto& inliers = epipolar_edges.E_inliers[0];
  auto sample_best = Tensor_<int, 1>{};
  auto estimator = NisterFivePointAlgorithm{};
  auto distance = EpipolarDistance{};
  {
    num_samples = 1000;
    err_thres = 1e-3;
    std::tie(E, inliers, sample_best) =
        ransac(M, un[0], un[1], estimator, distance, num_samples, err_thres);
    E.matrix() = E.matrix().normalized();

    epipolar_edges.E_inliers[0] = inliers;
    epipolar_edges.E_best_samples[0] = sample_best;
  }


  // Calculate the fundamental matrix.
  print_stage("Computing the fundamental matrix...");
  auto& F = epipolar_edges.F[0];
  {
    F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];

    epipolar_edges.F_num_samples[0] = 1000;
    epipolar_edges.F_noise = epipolar_edges.E_noise;
    epipolar_edges.F_inliers = epipolar_edges.E_inliers;
    epipolar_edges.F_best_samples = epipolar_edges.E_best_samples;
  }


  // Extract the two-view geometry.
  print_stage("Estimating the two-view geometry...");
  epipolar_edges.two_view_geometries = {
      estimate_two_view_geometry(M, un[0], un[1], E, inliers, sample_best)};
  auto& two_view_geometry = epipolar_edges.two_view_geometries.front();
  two_view_geometry.C1.K = views.cameras[0].K;
  two_view_geometry.C2.K = views.cameras[1].K;


  print_stage("Mapping feature GID to match GID...");
  const auto match_index = map_feature_gid_to_match_gid(epipolar_edges);


  print_stage("Populating the feature tracks...");
  const auto [feature_graph, components] =
      populate_feature_tracks(views, epipolar_edges);

  // Keep feature tracks of size 2 at least.
  print_stage("Checking the feature tracks...");
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


  print_stage("Check the SfM...");
  SARA_DEBUG << "camera_parameters =\n"
             << ba_problem.camera_parameters.matrix() << std::endl;
  SARA_DEBUG << "points =\n"
             << ba_problem.points_abs_coords_3d.matrix() << std::endl;

  // TODO: check the point reprojection errors.

  return 0;
}
