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
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/RANSAC/Utility.hpp>
#include <DO/Sara/Visualization.hpp>

#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


// =============================================================================
// Feature detection and matching.
//
// Detect or read SIFT keypoints.
auto get_keypoints(const Image<Rgb8>& image1, const Image<Rgb8>& image2,
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

  const auto [F, inliers, sample_best] = ransac(
      M, P1, P2, FEstimator{}, EpipolarDistance{}, num_samples, f_err_thres);

  return std::make_tuple(F, inliers, sample_best);
}

auto estimate_fundamental_matrix_v2(const KeypointList<OERegion, float>& keys1,
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
  const auto p1 = homogeneous(extract_centers(f1)).cast<double>();
  const auto p2 = homogeneous(extract_centers(f2)).cast<double>();

  // Get the list of point correspondences.
  const auto X = PointCorrespondenceList{M, p1, p2};

  const auto data_normalizer =
      std::make_optional(Normalizer<FundamentalMatrix>{X});

  auto inlier_predicate = InlierPredicate<EpipolarDistance>{};
  inlier_predicate.err_threshold = f_err_thres;

  const auto [F, inliers, sample_best] = ransac_v2(
      X, FEstimator{}, inlier_predicate, num_samples, data_normalizer);

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
  const auto h = int(max(image1.height(), image2.height()) * scale);

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

  SARA_DEBUG << "Number of inliers = " << inliers.flat_array().count()
             << std::endl;

  drawer.display_images();

  // Show the inliers.
  const auto num_matches = static_cast<int>(matches.size());
  for (auto i = 0; i < num_matches; ++i)
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
  Matrix<double, 3, L> proj_X = F.matrix() * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, L> proj_Y = F.matrix().transpose() * Y;
  proj_Y.array().rowwise() /= proj_Y.row(2).array();

  for (auto i = 0; i < L; ++i)
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


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

int sara_graphics_main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " image1 image2" << std::endl;
    return 1;
  }

  // Load images.
  print_stage("Loading images...");
  const auto images = std::array{imread<Rgb8>(argv[1]), imread<Rgb8>(argv[2])};

  // Use the following data structure to load images, keypoints, camera
  // parameters.
  print_stage("Computing keypoints...");
  const auto image_pyr_params = ImagePyramidParams(-1);
  auto keypoints = std::array{
      compute_sift_keypoints(images[0].convert<float>(), image_pyr_params),
      compute_sift_keypoints(images[1].convert<float>(), image_pyr_params)};

  print_stage("Matching keypoints...");
  const auto sift_nn_ratio = argc < 6 ? 0.6f : std::stof(argv[5]);
  const auto matches = match(keypoints[0], keypoints[1], sift_nn_ratio);


  print_stage("Estimating the fundamental matrix...");
  const auto num_samples = argc < 4 ? 200 : std::stoi(argv[3]);
  const auto f_err_thres = argc < 5 ? 1e-2 : std::stod(argv[4]);
  const auto [F, inliers, sample_best] = estimate_fundamental_matrix_v2(
      keypoints[0], keypoints[1], matches, num_samples, f_err_thres);


  print_stage("Inspecting the fundamental matrix estimation...");
  inspect_fundamental_matrix_estimation(images[0], images[1], matches, F,
                                        inliers, sample_best);

  return 0;
}
