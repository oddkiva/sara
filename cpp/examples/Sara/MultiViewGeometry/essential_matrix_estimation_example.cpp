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
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>

#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


using namespace std;
using namespace DO::Sara;


using EEstimator = NisterFivePointAlgorithm;


// Detect or read SIFT keypoints.
auto get_keypoints(const Image<Rgb8>& image1,
                   const Image<Rgb8>& image2,
                   const std::string& keys1_filepath,
                   const std::string& keys2_filepath,
                   KeypointList<OERegion, float>& keys1,
                   KeypointList<OERegion, float>& keys2)
{
  print_stage("Computing/Reading keypoints");

#ifdef COMPUTE_KEYPOINTS
  keys1 = compute_sift_keypoints(image1.convert<float>());
  keys2 = compute_sift_keypoints(image2.convert<float>());
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  const auto& [f1, d1] = keys1;
  const auto& [f2, d2] = keys2;

  write_keypoints(f1, d1, keys1_filepath);
  write_keypoints(f2, d2, keys2_filepath);
#else
  (void) image1;
  (void) image2;
  auto& [f1, d1] = keys1;
  auto& [f2, d2] = keys2;

  read_keypoints(f1, d1, keys1_filepath);
  read_keypoints(f2, d2, keys2_filepath);
#endif
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


  print_stage("Getting keypoints...");
  views.keypoints.resize(2 /* views */);
  get_keypoints(views.images[0], views.images[1],  //
                data_dir + "/" + "0000.key",       //
                data_dir + "/" + "0001.key",       //
                views.keypoints[0], views.keypoints[1]);


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
  auto estimator = EEstimator{};
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
      estimate_two_view_geometry(M, un[0], un[1], E, inliers, sample_best)
  };


  std::map<FeatureGID, int> feature_gids_to_3d_point_indices;


  // Populate the feature tracks.
  const auto [feature_graph, components] =
      populate_feature_tracks(views, epipolar_edges);

  // Keep feature tracks of size 2 at least.
  const auto feature_tracks = filter_feature_tracks(feature_graph, components);


  // Prepare the bundle adjustment problem formulation.
  //
  // 1. Count the number of 3D points.
  const auto num_points = static_cast<int>(feature_tracks.size());
  SARA_CHECK(num_points);

  // 2. Count the number of 2D observations.
  auto num_observations_per_points = std::vector<int>(num_points);
  std::transform(
      std::begin(feature_tracks), std::end(feature_tracks),
      std::begin(num_observations_per_points),
      [](const auto& track) { return static_cast<int>(track.size()); });

  const auto num_observations =
      std::accumulate(std::begin(num_observations_per_points),
                      std::end(num_observations_per_points), 0);
  SARA_CHECK(num_observations);

  // 3. Count the number of cameras, which should be equal to the number of
  //    images.
  auto image_ids = std::set<int>{};
  for (const auto& track : feature_tracks)
    for (const auto& f : track)
      image_ids.insert(f.image_id);

  const auto num_cameras = static_cast<int>(image_ids.size());
  SARA_CHECK(num_cameras);

  const auto num_parameters = 9 * num_cameras + 3 * num_points;
  SARA_CHECK(num_parameters);

  // 4. Transform the data for convenience.
  struct ObservationRef {
    FeatureGID gid;
    // TODO: needs the match_index;
    int camera_id;
    int point_id;
  };
  auto obs_refs = std::vector<ObservationRef>{};
  {
    obs_refs.reserve(num_observations);

    auto point_id = 0;
    for (const auto& track : feature_tracks)
    {
      for (const auto& f: track)
        obs_refs.push_back({f, f.image_id, point_id});
      ++point_id;
    }
  }

  // 5. Prepare the data for Ceres.
  auto observations = Tensor_<double, 2>{{num_observations, 2}};
  auto point_indices = std::vector<int>(num_observations);
  auto camera_indices = std::vector<int>(num_observations);
  auto parameters = std::vector<double>(num_parameters);
  for (int i = 0; i < num_observations; ++i)
  {
    const auto& ref = obs_refs[i];

    // Easy things first.
    point_indices[i] = ref.point_id;
    camera_indices[i] = ref.camera_id;

    // Initialize the 2D observations.
    const auto& image_id = ref.gid.image_id;
    const auto& local_id = ref.gid.local_id;
    const double x = un[image_id](local_id, 0);
    const double y = un[image_id](local_id, 1);
    observations(i, 0) = x;
    observations(i, 1) = y;

    // Initialize the 3D points.
    //parameters[9 * num_cameras + point_indices[i] + 0] =
  }

#ifdef SAVE_TWO_VIEW_GEOMETRY
  keep_cheiral_inliers_only(geometry, inliers);

  // Add the internal camera matrices to the camera.
  geometry.C1.K = K1;
  geometry.C2.K = K2;
  auto colors = extract_colors(image1, image2, geometry);
  save_to_hdf5(geometry, colors);
#endif


  // Inspect the fundamental matrix.
  print_stage("Inspecting the fundamental matrix estimation...");
  check_epipolar_constraints(views.images[0], views.images[1], F, matches,
                             sample_best, inliers,
                             /* display_step */ 20, /* wait_key */ true);

  return 0;
}
