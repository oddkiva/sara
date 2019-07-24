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

#include <DO/Sara/Core/MultiArray/DataTransformations.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>


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
using EEstimator = NisterFivePointAlgorithm;

void inspect_geometry(const TwoViewGeometry& g,
                      const Matrix<double, 3, EEstimator::num_points>& un1_s,
                      const Matrix<double, 3, EEstimator::num_points>& un2_s)
{
  const Matrix34d C1 = g.C1;
  const Matrix34d C2 = g.C2;

  SARA_DEBUG << "Camera matrices" << std::endl;
  SARA_DEBUG << "C1 =\n" << C1 << std::endl;
  SARA_DEBUG << "C2 =\n" << C2 << std::endl;

  SARA_DEBUG << "Triangulated points" << std::endl;
  const MatrixXd C1X = C1 * g.X;
  const MatrixXd C2X = C2 * g.X;

  SARA_DEBUG << "C1 * X =\n" << C1X << std::endl;
  SARA_DEBUG << "C2 * X =\n" << C2X << std::endl;

  SARA_DEBUG << "Comparison with normalized coordinates" << std::endl;
  SARA_DEBUG << "(C1 * X).hnormalized() =\n"
             << C1X.colwise().hnormalized() << std::endl;
  SARA_DEBUG << "u1n_s =\n" << un1_s.colwise().hnormalized() << std::endl;

  SARA_DEBUG << "(C2 * X).hnormalized() =\n"
             << C2X.colwise().hnormalized() << std::endl;
  SARA_DEBUG << "u2n_s.hnormalized() =\n"
             << un2_s.colwise().hnormalized() << std::endl;
  std::cout << std::endl;

  const double residual1 =
      (C1X.colwise().hnormalized() - un1_s.colwise().hnormalized()).norm() /
      (un1_s.colwise().hnormalized()).norm();
  const double residual2 =
      (C2X.colwise().hnormalized() - un2_s.colwise().hnormalized()).norm() /
      (un1_s.colwise().hnormalized()).norm();
  SARA_DEBUG << "Residual 1 = " << residual1 << std::endl;
  SARA_DEBUG << "Residual 2 = " << residual2 << std::endl;

  SARA_DEBUG << "Cheirality =\n" << g.cheirality << std::endl;
}

auto estimate_two_view_geometry(const TensorView_<int, 2>& M,
                                const TensorView_<double, 2>& un1,
                                const TensorView_<double, 2>& un2,
                                const EssentialMatrix& E,
                                const TensorView_<bool, 1>& inliers,
                                const TensorView_<int, 1>& sample_best)
{
  // Visualize the best sample drawn by RANSAC.
  constexpr auto L = EEstimator::num_points;
  const auto sample_best_reshaped = sample_best.reshape(Vector2i{1, L});
  const auto I = to_point_indices(sample_best_reshaped, M);
  const auto un_s = to_coordinates(I, un1, un2).transpose({0, 2, 1, 3});
  const Matrix<double, 3, L> un1_s = un_s[0][0].colmajor_view().matrix();
  const Matrix<double, 3, L> un2_s = un_s[0][1].colmajor_view().matrix();

  const auto candidate_motions = extract_relative_motion_horn(E);

  // Triangulate the points from the best samples and calculate their
  // cheirality.
  auto geometries = std::vector<TwoViewGeometry>{};
  std::transform(std::begin(candidate_motions), std::end(candidate_motions),
                 std::back_inserter(geometries), [&](const Motion& m) {
                   return two_view_geometry(m, un1_s, un2_s);
                 });
#ifdef DEBUG
  // Check the cheirality.
  for (const auto& g: geometries)
    inspect_geometry(g, un1_s, un2_s);
#endif

  // Find the best geometry, i.e., the one with the high cheirality degree.
  const auto best_geom =
      std::max_element(std::begin(geometries), std::end(geometries),
                       [](const auto& g1, const auto& g2) {
                         return g1.cheirality.count() < g2.cheirality.count();
                       });


  const auto cheiral_degree = best_geom->cheirality.count();
  if (cheiral_degree == 0)
    throw std::runtime_error{"The cheirality degree can't be zero!"};

  inspect_geometry(*best_geom, un1_s, un2_s);
  // Data transformations.
  const Matrix34d P1 = best_geom->C1;
  const Matrix34d P2 = best_geom->C2;

  // Extract the matched coordinates.
  const auto card_M = M.size(0);
  const auto mindices = range(card_M);
  const auto card_M_filtered = mindices.size(0);
  SARA_CHECK(card_M_filtered);

  auto coords_matched = Tensor_<double, 3>{{2, card_M_filtered, 3}};
  auto un1_matched_mat = coords_matched[0].colmajor_view().matrix();
  auto un2_matched_mat = coords_matched[1].colmajor_view().matrix();
  {
    const auto un1_mat = un1.colmajor_view().matrix();
    const auto un2_mat = un2.colmajor_view().matrix();
    std::for_each(std::begin(mindices), std::end(mindices), [&](int m) {
      un1_matched_mat.col(m) = un1_mat.col(M(m, 0));
      un2_matched_mat.col(m) = un2_mat.col(M(m, 1));
    });
  }
#ifdef DEBUG
  SARA_DEBUG << "un1_matched_mat =\n"
             << un1_matched_mat.leftCols(10) << std::endl;
  SARA_DEBUG << "un2_matched_mat =\n"
             << un2_matched_mat.leftCols(10) << std::endl;
#endif

  SARA_DEBUG << "Triangulating all matches..." << std::endl;
  auto X = triangulate_linear_eigen(P1, P2, un1_matched_mat, un2_matched_mat);
  SARA_CHECK(X.cols());
  SARA_CHECK(inliers.flat_array().count());

  SARA_DEBUG << "Calculating cheirality..." << std::endl;
  auto cheirality = relative_motion_cheirality_predicate(X, P2);

  SARA_DEBUG << "Keep cheiral inliers..." << std::endl;
  const auto X_cheiral =
      range(X.cols())                                                 //
      | filtered([&](int i) { return cheirality(i) && inliers(i); })  //
      | transformed([&](int i) -> Vector4d { return X.col(i); });
  SARA_CHECK(X_cheiral.size());

  auto complete_geom = *best_geom;
  {
    complete_geom.X.resize(4, X_cheiral.size(0));
    for (int i = 0; i < X_cheiral.size(0); ++i)
      complete_geom.X.col(i) = X_cheiral(i);

    complete_geom.cheirality =
        relative_motion_cheirality_predicate(complete_geom.X, P2);
  }

  SARA_DEBUG << "complete_geom.X =\n"
             << complete_geom.X.leftCols(10) << std::endl;
  SARA_DEBUG << "complete_geom.cheirality = "
             << complete_geom.cheirality.count() << std::endl;

  return complete_geom;
}

auto extract_colors(const Image<Rgb8>& image1,             //
                    const Image<Rgb8>& image2,             //
                    const TwoViewGeometry& complete_geom)  //
{
  const int num_points = complete_geom.X.cols();
  const auto indices = range(num_points);
  auto colors = Tensor_<double, 2>{num_points, 3};

  const auto I1d = image1.convert<Rgb64f>();
  const auto I2d = image2.convert<Rgb64f>();

  const auto P1 = complete_geom.C1.matrix();
  const auto P2 = complete_geom.C2.matrix();

  const MatrixXd u1 = (P1 * complete_geom.X).colwise().hnormalized();
  const MatrixXd u2 = (P2 * complete_geom.X).colwise().hnormalized();

  auto colors_mat = colors.matrix();
  std::for_each(std::begin(indices), std::end(indices), [&](int i) {
    Vector2d u1_i = u1.col(i);
    Vector2d u2_i = u2.col(i);
    colors_mat.row(i) =
        0.5 * (interpolate(I1d, u1_i) + interpolate(I2d, u2_i)).transpose();
  });

  return colors;
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

    if (i % 50 == 0)
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
  constexpr auto L = EEstimator::num_points;
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

  for (size_t i = 0; i < L; ++i)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[sample_best(i)], Red8, true);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(), Yellow8, 1);
    drawer.draw_line_from_eqn(0, proj_Y.col(i).cast<float>(), Yellow8, 1);
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
  const auto matches = compute_matches(keys1, keys2);


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
      v2::ransac(M, un1, un2, e_estimator, distance, num_samples, e_err_thres);


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


  // Add the internal camera matrices to the camera.
  geometry.C1.K = K1;
  geometry.C2.K = K2;
  auto colors = extract_colors(image1, image2, geometry);
  save_to_hdf5(geometry, colors);


  // Inspect the fundamental matrix.
  print_stage("Inspecting the fundamental matrix estimation...");
  inspect_fundamental_matrix_estimation(image1, image2, matches, F, inliers,
                                        sample_best);

  return 0;
}
