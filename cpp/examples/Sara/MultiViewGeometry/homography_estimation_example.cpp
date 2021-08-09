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
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/Visualization.hpp>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


// =============================================================================
// Feature detection and matching.
//
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

// Descriptor-based matching.
//
// TODO: by default a feature matcher should return a tensor containing pair of
// indices instead. While becoming trickier to use, it is more powerful for data
// manipulation.
//
// Convert a set of matches to a tensor.
auto compute_matches(const KeypointList<OERegion, float>& keys1,
                     const KeypointList<OERegion, float>& keys2,
                     float lowe_ratio = 0.6f)
{
  print_stage("Computing Matches");
  AnnMatcher matcher{keys1, keys2, lowe_ratio};

  const auto matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;

  return matches;
}


// =============================================================================
// Robust estimation using RANSAC.
//
auto estimate_homography(const KeypointList<OERegion, float>& keys1,
                         const KeypointList<OERegion, float>& keys2,
                         const vector<Match>& matches, int num_samples,
                         double h_err_thres)
{
  // ==========================================================================
  // Normalize the points.
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = extract_centers(f1).cast<double>();
  const auto p2 = extract_centers(f2).cast<double>();

  // Work in homogeneous coordinates please.
  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  const auto M = to_tensor(matches);

  // Generate random samples for RANSAC.
  auto distance = SymmetricTransferError{};

  // const auto [H, num_inliers, sample_best] = ransac(
  //     M, P1, P2, FourPointAlgorithm{}, distance, num_samples, h_err_thres);
  const auto [H, inliers, sample_best] = ransac(
      M, P1, P2, FourPointAlgorithm{}, distance, num_samples, h_err_thres);

  return std::make_tuple(H, inliers, sample_best);
}


// =============================================================================
// Visual inspection.
//
void inspect_homography_estimation(const Image<Rgb8>& image1,
                                   const Image<Rgb8>& image2,  //
                                   const vector<Match>& matches,
                                   const Matrix3d& H,
                                   const Tensor_<bool, 1>& inliers,
                                   const Tensor_<int, 1>& sample_best)
{
  // ==========================================================================
  // Setup the visualization.
  const auto scale = .25f;
  const auto w = static_cast<int>((image1.width() + image2.width()) * scale);
  const auto h =
      static_cast<int>(max(image1.height(), image2.height()) * scale);

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

  SARA_DEBUG << "Number of inliers = " << inliers.flat_array().count()
             << std::endl;

  drawer.display_images();

  // Show the inliers.
  for (auto i = 0u; i < matches.size(); ++i)
    if (inliers(i))
      drawer.draw_match(matches[i], Blue8, false);

  // Extract the best sampled points found by RANSAC and calculate their
  // projections in the images.
  constexpr auto L = FourPointAlgorithm::num_points;
  auto X = Matrix<double, 3, L>{};
  auto Y = Matrix<double, 3, L>{};
  for (auto i = 0; i < L; ++i)
  {
    const auto& mi = matches[sample_best(i)];
    X.col(i) = mi.x_pos().cast<double>().homogeneous();
    Y.col(i) = mi.y_pos().cast<double>().homogeneous();
  };

  // Project X to the right image.
  Matrix<double, 3, L> proj_X = H * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, L> proj_Y = H.inverse() * Y;
  proj_Y.array().rowwise() /= proj_Y.row(2).array();

  for (size_t i = 0; i < L; ++i)
  {
    // Draw the corresponding projected points.
    drawer.draw_point(1, proj_X.col(i).head(2).cast<float>(), Magenta8, 7);
    drawer.draw_point(0, proj_Y.col(i).head(2).cast<float>(), Magenta8, 7);
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[sample_best(i)], Red8, true);

  }

  get_key();
  close_window();
}


GRAPHICS_MAIN()
{
  // Load images.
  print_stage("Loading images...");
#ifdef __APPLE__
  const auto data_dir =
      "/Users/david/Desktop/Datasets/sfm/castle_int"s;
#else
  const auto data_dir =
      "/home/david/Desktop/Datasets/sfm/castle_int"s;
#endif
  const auto file1 = "0000.png";
  const auto file2 = "0001.png";

  const auto image1 = imread<Rgb8>(data_dir + "/" + file1);
  const auto image2 = imread<Rgb8>(data_dir + "/" + file2);

  print_stage("Get keypoints...");
  auto keys1 = KeypointList<OERegion, float>{};
  auto keys2 = KeypointList<OERegion, float>{};
  get_keypoints(image1, image2,               //
                data_dir + "/" + "0000.key",  //
                data_dir + "/" + "0001.key",  //
                keys1, keys2);

  print_stage("Match keypoints...");
  const auto matches = compute_matches(keys1, keys2);

  print_stage("Estimate the principal homography...");
  const auto num_samples = 1000;
  const auto h_err_thres = 1.;
  const auto [H, inliers, sample_best] =
      estimate_homography(keys1, keys2, matches, num_samples, h_err_thres);

  print_stage("Inspect the homography estimation...");
  inspect_homography_estimation(image1, image2, matches, H, inliers,
                                sample_best);

  return 0;
}
