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
  // Load images.
  print_stage("Loading images...");
#ifdef __APPLE__
  const auto data_dir =
      std::string{"/Users/david/Desktop/Datasets/sfm/castle_int"};
#else
  const auto data_dir =
      std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
#endif
  const auto file1 = "0000.png";
  const auto file2 = "0001.png";

  const auto image1 = imread<Rgb8>(data_dir + "/" + file1);
  const auto image2 = imread<Rgb8>(data_dir + "/" + file2);


  print_stage("Loading the internal camera matrices...");
  const Matrix3d K1 =
      read_internal_camera_parameters(data_dir + "/" + "0000.png.K")
          .cast<double>();
  const Matrix3d K2 =
      read_internal_camera_parameters(data_dir + "/" + "0001.png.K")
          .cast<double>();
  const Matrix3d K1_inv = K1.inverse();
  const Matrix3d K2_inv = K2.inverse();


  print_stage("Getting keypoints...");
  auto keys1 = KeypointList<OERegion, float>{};
  auto keys2 = KeypointList<OERegion, float>{};
  get_keypoints(image1, image2,               //
                data_dir + "/" + "0000.key",  //
                data_dir + "/" + "0001.key",  //
                keys1, keys2);


  print_stage("Matching keypoints...");
  const auto matches = match(keys1, keys2);


  print_stage("Performing data transformations...");
  // Tensors of image coordinates.
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto u1 = homogeneous(extract_centers(f1)).cast<double>();
  const auto u2 = homogeneous(extract_centers(f2)).cast<double>();
  // Tensors of camera coordinates.
  const auto un1 = apply_transform(K1_inv, u1);
  const auto un2 = apply_transform(K2_inv, u2);
  static_assert(std::is_same_v<decltype(un1), const Tensor_<double,2>>);
  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  const auto M = to_tensor(matches);


  print_stage("Estimating the essential matrix...");
  const auto num_samples = 1000;
  const auto e_err_thres = 1e-3;
  auto e_estimator = EEstimator{};
  auto distance = EpipolarDistance{};
  const auto [E_, inliers, sample_best] =
      ransac(M, un1, un2, e_estimator, distance, num_samples, e_err_thres);


  // Calculate the fundamental matrix.
  print_stage("Computing the fundamental matrix...");
  EssentialMatrix E = E_;
  E.matrix() = E_.matrix().normalized();
  auto F = FundamentalMatrix{};
  F.matrix() = K2_inv.transpose() * E.matrix() * K1_inv;


  // Extract the two-view geometry.
  print_stage("Estimating the two-view geometry...");
  auto geometry =
      estimate_two_view_geometry(M, un1, un2, E, inliers, sample_best);

  keep_cheiral_inliers_only(geometry, inliers);

  // Add the internal camera matrices to the camera.
  geometry.C1.K = K1;
  geometry.C2.K = K2;
  auto colors = extract_colors(image1, image2, geometry);
  save_to_hdf5(geometry, colors);


  // Inspect the fundamental matrix.
  print_stage("Inspecting the fundamental matrix estimation...");
  check_epipolar_constraints(image1, image2, F, matches, sample_best, inliers,
                             /* display_step */ 20, /* wait_key */ true);

  return 0;
}
