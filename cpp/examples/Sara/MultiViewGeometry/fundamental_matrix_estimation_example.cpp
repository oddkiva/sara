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

#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


using namespace std;
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
using FEstimator = EightPointAlgorithm;

auto estimate_fundamental_matrix(const KeypointList<OERegion, float>& keys1,
                                 const KeypointList<OERegion, float>& keys2,
                                 const vector<Match>& matches,  //
                                 int num_samples, double f_err_thres)
{
  // Transform matches to an array of indices.
  const auto M = to_tensor(matches);

  // ==========================================================================
  // Image coordinates.
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = extract_centers(f1).cast<double>();
  const auto p2 = extract_centers(f2).cast<double>();

  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  const auto [F, inliers, sample_best] = v2::ransac(
      M, P1, P2, FEstimator{}, EpipolarDistance{}, num_samples, f_err_thres);

  return std::make_tuple(F, inliers, sample_best);
}


// =============================================================================
// Visual inspection.
//
void inspect_fundamental_matrix_estimation(const Image<Rgb8>& image1,
                                           const Image<Rgb8>& image2,  //
                                           const vector<Match>& matches,
                                           const FundamentalMatrix& F,
                                           const Tensor_<bool, 1>& inliers,
                                           const Tensor_<int, 1>& sample_best)
{
  // ==========================================================================
  // Setup the visualization.
  const auto scale = .25f;
  const auto w = int((image1.width() + image2.width()) * scale);
  const auto h = max(image1.height(), image2.height()) * scale;

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

  SARA_DEBUG << "Number of inliers = " << inliers.flat_array().count()
             << std::endl;

  drawer.display_images();

  // Show the inliers.
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (!inliers(i))
      continue;

    if (i % 100 == 0)
    {
      drawer.draw_match(matches[i], Blue8, false);

      const Vector3d X1 = matches[i].x_pos().cast<double>().homogeneous();
      const Vector3d X2 = matches[i].y_pos().cast<double>().homogeneous();

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
    }
  }

  // Extract the best sampled points found by RANSAC and calculate their
  // projections in the images.
  constexpr auto L = EightPointAlgorithm::num_points;
  auto X = Matrix<double, 3, L>{};
  auto Y = Matrix<double, 3, L>{};
  for (auto i = 0; i < L; ++i)
  {
    const auto& mi = matches[sample_best(i)];
    X.col(i) = mi.x_pos().cast<double>().homogeneous();
    Y.col(i) = mi.y_pos().cast<double>().homogeneous();
  };

  // Project X to the right image.
  Matrix<double, 3, 8> proj_X = F.matrix() * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, 8> proj_Y = F.matrix().transpose() * Y;
  proj_Y.array().rowwise() /= proj_Y.row(2).array();

  for (size_t i = 0; i < L; ++i)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[sample_best(i)], Red8, true);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_Y.col(i).cast<float>(), Magenta8, 1);
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
      std::string{"/Users/david/Desktop/Datasets/sfm/castle_int"};
#else
  const auto data_dir =
      std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
#endif
  const auto file1 = "0000.png";
  const auto file2 = "0001.png";

  const auto image1 = imread<Rgb8>(data_dir + "/" + file1);
  const auto image2 = imread<Rgb8>(data_dir + "/" + file2);

  print_stage("Getting keypoints...");
  auto keys1 = KeypointList<OERegion, float>{};
  auto keys2 = KeypointList<OERegion, float>{};
  get_keypoints(image1, image2,               //
                data_dir + "/" + "0000.key",  //
                data_dir + "/" + "0001.key",  //
                keys1, keys2);

  print_stage("Matching keypoints...");
  const auto matches = compute_matches(keys1, keys2);

  print_stage("Estimating the fundamental matrix...");
  const auto num_samples = 1000;
  const double f_err_thres = 1e-2;
  const auto [F, inliers, sample_best] = estimate_fundamental_matrix(
      keys1, keys2, matches, num_samples, f_err_thres);


  print_stage("Inspecting the fundamental matrix estimation...");
  inspect_fundamental_matrix_estimation(image1, image2, matches, F, inliers,
                                        sample_best);

  return 0;
}
