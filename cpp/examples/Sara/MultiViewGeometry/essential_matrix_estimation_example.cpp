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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/MultiViewGeometry/BundleAdjustmentProblem.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>

#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


using namespace std;
using namespace DO::Sara;


GRAPHICS_MAIN()
{
  // Use the following data structure to load images, keypoints, camera
  // parameters.
  auto views = ViewAttributes{};

  // Load images.
  print_stage("Loading images...");
  const auto data_dir =
#ifdef __APPLE__
      std::string{"/Users/david/Desktop/Datasets/sfm/castle_int"};
#else
      std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
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

  // Filter the 3D points.
  auto& two_view_geometry = epipolar_edges.two_view_geometries.front();
  keep_cheiral_inliers_only(two_view_geometry, inliers);

  // Add the internal camera matrices to the camera.
  two_view_geometry.C1.K = views.cameras[0].K;
  two_view_geometry.C2.K = views.cameras[1].K;
  auto colors =
      extract_colors(views.images[0], views.images[1], two_view_geometry);
  save_to_hdf5(two_view_geometry, colors);

  // Inspect the fundamental matrix.
  print_stage("Inspecting the fundamental matrix estimation...");
  check_epipolar_constraints(views.images[0], views.images[1], F, matches,
                             sample_best, inliers,
                             /* display_step */ 20, /* wait_key */ true);

  return 0;
}
