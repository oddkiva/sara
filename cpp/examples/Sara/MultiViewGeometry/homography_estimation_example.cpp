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
#include <DO/Sara/RANSAC/Utility.hpp>
#include <DO/Sara/Visualization.hpp>

#include <filesystem>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


// =============================================================================
// Feature detection and matching.
//
// Detect or read SIFT keypoints.
auto get_keypoints(const Image<Rgb8>& image1, const Image<Rgb8>& image2,
                   [[maybe_unused]] const std::string& keys1_filepath,
                   [[maybe_unused]] const std::string& keys2_filepath,
                   KeypointList<OERegion, float>& keys1,
                   KeypointList<OERegion, float>& keys2)
{
  print_stage("Computing/Reading keypoints");

  namespace fs = std::filesystem;

#ifdef FIXME
  const auto load_or_compute_keys = [](auto& keys, const auto& keys_filepath,
                                       const auto& image) {
    if (!fs::exists(keys_filepath))
    {
      keys = compute_sift_keypoints(image.template convert<float>());
      const auto& [f, d] = keys;
      write_keypoints(f, d, keys_filepath);
    }
    else
    {
      auto& [f, d] = keys;
      read_keypoints(f, d, keys_filepath);
    }
  };

  load_or_compute_keys(keys1, keys1_filepath, image1);
  load_or_compute_keys(keys2, keys2_filepath, image2);
#else
  keys1 = compute_sift_keypoints(image1.template convert<float>());
  keys2 = compute_sift_keypoints(image2.template convert<float>());
#endif

  cout << "Image 1: " << features(keys1).size() << " keypoints" << endl;
  cout << "Image 2: " << features(keys2).size() << " keypoints" << endl;
}

// Descriptor-based matching.
//
// TODO: by default a feature matcher should return a tensor containing pair
// of indices instead. While becoming trickier to use, it is more powerful for
// data manipulation.
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

auto estimate_homography_v2(const KeypointList<OERegion, float>& keys1,
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

  const auto M = to_tensor(matches);
  const auto to_data = [](const TensorView_<int, 2>& M,
                          const TensorView_<double, 2>& p1,
                          const TensorView_<double, 2>& p2) {
    auto X = Tensor_<double, 3>{2, M.size(0), p1.size(1)};
    auto p1_mat = p1.matrix();
    auto p2_mat = p2.matrix();
    auto p1_matched = X[0].matrix();
    auto p2_matched = X[1].matrix();
    for (auto m = 0; m < M.size(0); ++m)
    {
      const auto& i1 = M(m, 0);
      const auto& i2 = M(m, 1);

      p1_matched.row(m) = p1_mat.row(i1);
      p2_matched.row(m) = p2_mat.row(i2);
    }
    return X;
  };

  const auto X = to_data(M, p1, p2);
  std::cout << "to_data: OK\n";

  const auto data_normalizer = Normalizer<Homography>{X[0], X[1]};
  std::cout << "data_normalizer: OK\n";

  auto Xn = Tensor_<double, 3>{X.sizes()};
  Xn[0] = apply_transform(data_normalizer.T1, X[0]);
  Xn[1] = apply_transform(data_normalizer.T2, X[1]);
  std::cout << "data normalization applied: OK\n";

  const auto num_data_points = X.size(1);
  const auto minimal_index_subsets = random_samples(1000, 4, X.size(1));
  std::cout << "random minimal subsets: OK\n";

  // The simplest and most natural implementation.
  auto p1_sampled = from_index_to_point(minimal_index_subsets, Xn[0]);
  auto p2_sampled = from_index_to_point(minimal_index_subsets, Xn[1]);
  std::cout << "match coordinates: OK\n";
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

  for (auto i = 0; i < L; ++i)
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
  const auto data_dir = "/Users/david/Desktop/Datasets/sfm/castle_int"s;
#else
  const auto data_dir = "/home/david/Desktop/Datasets/sfm/castle_int"s;
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

#ifdef OLD
  print_stage("Estimate the principal homography...");
  const auto num_samples = 1000;
  const auto h_err_thres = 1.;
  const auto [H, inliers, sample_best] =
      estimate_homography(keys1, keys2, matches, num_samples, h_err_thres);

  print_stage("Inspect the homography estimation...");
  inspect_homography_estimation(image1, image2, matches, H, inliers,
                                sample_best);
#else
  print_stage("Estimate the principal homography...");
  const auto num_samples = 1000;
  const auto h_err_thres = 1.;
  estimate_homography_v2(keys1, keys2, matches, num_samples, h_err_thres);

  // print_stage("Inspect the homography estimation...");
  // inspect_homography_estimation(image1, image2, matches, H, inliers,
  //                               sample_best);
#endif

  return 0;
}
