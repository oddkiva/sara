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
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <tinyply-2.2/source/tinyply.h>


using namespace std;
using namespace DO::Sara;


#ifdef __APPLE__
const auto data_dir =
    std::string{"/Users/david/Desktop/Datasets/sfm/castle_int"};
#else
const auto data_dir =
    std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
#endif
const auto file1 = "0000.png";
const auto file2 = "0001.png";


// =============================================================================
// Miscellaneous utilities.
//
auto print_3d_array(const TensorView_<float, 3>& x)
{
  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << fixed << x(i,j,k);
        if (k != x.size(2) - 1)
          cout << ", ";
      }
      cout << "]";

      if (j != x.size(1) - 1)
        cout << ", ";
      else
        cout << "]";
    }

    if (i != x.size(0) - 1)
      cout << ",\n ";
  }
  cout << "]" << endl;
}

// =============================================================================
// I/O utilities.
//
auto read_internal_camera_parameters(const std::string& filepath) -> Matrix3f
{
  std::ifstream file{filepath};
  if (!file)
    throw std::runtime_error{"File " + filepath + "does not exist!"};

  Matrix3f K;
  file >> K;

  return K;
}


// =============================================================================
// Feature detection and matching.
//
auto get_keypoints(KeypointList<OERegion, float>& keys1,
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

  write_keypoints(f1, d1, data_dir + "/" + "0000.key");
  write_keypoints(f2, d2, data_dir + "/" + "0001.key");
#else
  auto& [f1, d1] = keys1;
  auto& [f2, d2] = keys2;

  read_keypoints(f1, d1, data_dir + "/" + "0000.key");
  read_keypoints(f2, d2, data_dir + "/" + "0001.key");
#endif
}

// TODO: by default a feature matcher should just return a tensor. It is
// more cryptic but more powerful to manipulate data.
//
// Convert a set of matches to a tensor.
auto compute_matches(const KeypointList<OERegion, float>& keys1,
                     const KeypointList<OERegion, float>& keys2)
{
  print_stage("Computing Matches");
  AnnMatcher matcher{keys1, keys2, 0.6f};

  const auto matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;

  return matches;
}


using FEstimator = EightPointAlgorithm;
using EEstimator = NisterFivePointAlgorithm;
auto f_estimator = FEstimator{};
auto e_estimator = EEstimator{};

inline auto epipolar_distance(const Matrix3d& F, const Vector3d& X, const Vector3d& Y)
{
  return std::abs(Y.transpose() * F * X);
};

const double e_err_thresh = 1e-3;
const double f_err_thresh = 1e-2;

// Count the inliers.
template <typename Distance>
auto count_inliers(const Matrix3d& F, const TensorView_<int, 2>& M,
                   const TensorView_<double, 2>& P1,
                   const TensorView_<double, 2>& P2,
                   Distance distance,
                   double thres)
{
  const auto P1mat = P1.colmajor_view().matrix();
  const auto P2mat = P2.colmajor_view().matrix();
  auto num_inliers = 0;
  for (auto m = 0; m < M.size(0); ++m)
  {
    const auto i = M(m, 0);
    const auto j = M(m, 1);

    const Vector3d xi = P1mat.col(i);
    const Vector3d yj = P2mat.col(j);

    // inlier predicate.
    if (distance(F, xi, yj) < thres)
      ++num_inliers;
  }

  return num_inliers;
};

// =============================================================================
// Multiview geometry estimation.
//
#ifdef DEBUG
void estimate_homography_old(const Image<Rgb8>& image1,
                             const Image<Rgb8>& image2,
                             const KeypointList<OERegion, float>& keys1,
                             const KeypointList<OERegion, float>& keys2,
                             const vector<Match>& matches)
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

  // ==========================================================================
  // Normalize the points.
  const auto to_double = [](const float& src) { return double(src); };
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = extract_centers(f1).cwise_transform(to_double);
  const auto p2 = extract_centers(f2).cwise_transform(to_double);

  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  const auto normalizer = Normalizer<Homography>{P1, P2};

  const auto [P1n, P2n] = normalizer.normalize(P1, P2);

  const auto M = to_tensor(matches);

  // Generate random samples for RANSAC.
  constexpr auto N = 1000;
  constexpr auto L = FourPointAlgorithm::num_points;
  const auto S = random_samples(N, L, int(matches.size()));

  const auto I = to_point_indices(S, M);
  const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  const auto h_estimator = FourPointAlgorithm{};

  auto distance = [](const Matrix3d& H, const Vector3d& x,
                     const Vector3d& y) -> double {
    return ((H * x).hnormalized() - y.hnormalized()).norm() +
           ((H.inverse() * y).hnormalized() - x.hnormalized()).norm();
  };

  for (auto n = 0; n < N; ++n)
  {
    // Extract the point
    const Matrix<double, 2, L> x = p[n][0].colmajor_view().matrix();
    const Matrix<double, 2, L> y = p[n][1].colmajor_view().matrix();

    const Matrix<double, 3, L> X = P[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Y = P[n][1].colmajor_view().matrix();

    const Matrix<double, 3, L> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Yn = Pn[n][1].colmajor_view().matrix();

    // 4-point algorithm
    auto [H] = h_estimator(Xn, Yn);
    std::cout << "Before denormalization..." << std::endl;
    std::cout << H.matrix() << std::endl;

    // Unnormalize the homography matrix.
    H.matrix() = normalizer.denormalize(H);

    std::cout << "Check H..." << std::endl;
    std::cout << H.matrix() << std::endl;

    MatrixXd HX = H.matrix() * X;
    HX.array().rowwise() /= HX.row(2).array();

    std::cout << "HX" << std::endl;
    std::cout << HX << std::endl << std::endl;

    std::cout << "Y" << std::endl;
    std::cout << Y << std::endl << std::endl;

    std::cout << "Algebraic errors:" << std::endl;
    for (int i = 0; i < 4; ++i)
      std::cout << (HX.col(i) - Y.col(i)).norm() << std::endl;
    std::cout << std::endl;

    // Display the result.
    drawer.display_images();

    for (size_t i = 0; i < matches.size(); ++i)
    {
      const Vector3d X1 = matches[i].x_pos().cast<double>().homogeneous();
      const Vector3d X2 = matches[i].y_pos().cast<double>().homogeneous();

      const Vector2d HX1 = (H.matrix() * X1).hnormalized();
      const Vector2d HX2 = (H.matrix().inverse() * X2).hnormalized();

      if (distance(H, X1, X2) < 1.)
      {
        drawer.draw_match(matches[i], Blue8, false);
        drawer.draw_point(0, HX2.cast<float>(), Cyan8, 5);
        drawer.draw_point(1, HX1.cast<float>(), Cyan8, 5);
      }
    };
    for (size_t i = 0; i < 4; ++i)
    {
      drawer.draw_match(matches[S(n, i)], Red8, true);

      drawer.draw_point(0, x.col(i).cast<float>(), Magenta8, 5);
      drawer.draw_point(1, y.col(i).cast<float>(), Magenta8, 5);

      drawer.draw_point(0, X.col(i).hnormalized().cast<float>(), Magenta8, 5);
      drawer.draw_point(1, Y.col(i).hnormalized().cast<float>(), Magenta8, 5);

      drawer.draw_point(1, HX.col(i).hnormalized().cast<float>(), Blue8, 5);
    }
  }
}

void estimate_fundamental_matrix_old(const Image<Rgb8>& image1,
                                     const Image<Rgb8>& image2,
                                     const KeypointList<OERegion, float>& keys1,
                                     const KeypointList<OERegion, float>& keys2,
                                     const vector<Match>& matches)
{
  // ==========================================================================
  // Setup the visualization.
  const auto scale = 0.25f;
  const auto w = int((image1.width() + image2.width()) * scale);
  const auto h = max(image1.height(), image2.height()) * scale;

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);


  // Transform matches to an array of indices.
  const auto M = to_tensor(matches);

  // ==========================================================================
  // Image coordinates.
  const auto to_double = [](const float& src) { return double(src); };
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = extract_centers(f1).cwise_transform(to_double);
  const auto p2 = extract_centers(f2).cwise_transform(to_double);

  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  // Normalization transformation.
  const auto T1 = compute_normalizer(P1);
  const auto T2 = compute_normalizer(P2);

  // Normalized image coordinates.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);

  SARA_DEBUG << "P1n =\n" << P1n.matrix().topRows(10) << std::endl;
  SARA_DEBUG << "P2n =\n" << P2n.matrix().topRows(10) << std::endl;

  // ==========================================================================
  // Generate random samples for RANSAC.
  constexpr auto N = 1000;
  constexpr auto L = EightPointAlgorithm::num_points;
  const auto S = random_samples(N, L, M.size(0));
  SARA_DEBUG << "L = " << L << std::endl;

  // M = list of matches.
  const auto card_M = M.size(0);

  SARA_DEBUG << "M =\n" << M.matrix().topRows(10) << std::endl;
  SARA_DEBUG << "card_M = " << card_M << std::endl;

  if (card_M < EightPointAlgorithm::num_points)
    throw std::runtime_error{"Not enough matches!"};


  // Remap each match index 'm' to a pair of point indices '(x, y)'.
  const auto I = to_point_indices(S, M);

  // Remap each pair of point indices to a pair of point coordinates.
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});

  // Normalized coordinates.
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  auto F_best = FundamentalMatrix{};
  auto num_inliers_best = 0;
  auto subset_best = 0;

  for (auto n = 0; n < N; ++n)
  {
    // Extract the corresponding points.
    const Matrix<double, 3, 8> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, 8> Yn = Pn[n][1].colmajor_view().matrix();

    // Estimate the fundamental matrix.
    auto [F] = f_estimator(Xn, Yn);

    // Unnormalize the fundamental matrix.
    F.matrix() = T2.transpose() * F.matrix().normalized() * T1;
    F.matrix() = F.matrix().normalized();

    const auto num_inliers =
        count_inliers(F, M, P1, P2, epipolar_distance, f_err_thresh);

    if (num_inliers > num_inliers_best)
    {
      num_inliers_best = num_inliers;
      F_best = F;
      subset_best = n;
    }
  }

  // Display the result.
  const auto& F = F_best;
  const auto& n = subset_best;

  // Extract the points.
  const Matrix<double, 3, 8> X = P[n][0].colmajor_view().matrix();
  const Matrix<double, 3, 8> Y = P[n][1].colmajor_view().matrix();

  SARA_CHECK(F);
  SARA_CHECK(num_inliers_best);
  SARA_CHECK(n);
  SARA_CHECK(matches.size());

  // Project X to the right image.
  Matrix<double, 3, 8> proj_X = F.matrix() * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, 8> proj_Y = F.matrix().transpose() * Y;
  proj_Y.array().rowwise() /= proj_Y.row(2).array();


  drawer.display_images();

  for (size_t i = 0; i < 8; ++i)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[S(n, i)], Red8, true);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_Y.col(i).cast<float>(), Magenta8, 1);
  }
  get_key();

  for (size_t i = 0; i < matches.size(); ++i)
  {
    const Vector3d X1 = matches[i].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = matches[i].y_pos().cast<double>().homogeneous();

    if (epipolar_distance(F, X1, X2) > f_err_thresh)
      continue;

    if (i % 100 == 0)
    {
      drawer.draw_match(matches[i], Blue8, false);

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
    }
  }

  get_key();
  close_window();
}

void estimate_essential_matrix_old(const Image<Rgb8>& image1,
                                   const Image<Rgb8>& image2,
                                   const Matrix3f& K1f, const Matrix3f& K2f,
                                   const KeypointList<OERegion, float>& keys1,
                                   const KeypointList<OERegion, float>& keys2,
                                   const vector<Match>& matches)
{
  // ==========================================================================
  // Setup the visualization.
  const auto scale = 0.25f;
  const auto w = int((image1.width() + image2.width()) * scale);
  const auto h = max(image1.height(), image2.height()) * scale;

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);


  // ==========================================================================
  // Image coordinates.
  const auto to_double = [](const float& src) { return double(src); };
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = homogeneous(extract_centers(f1).cwise_transform(to_double));
  const auto p2 = homogeneous(extract_centers(f2).cwise_transform(to_double));

  // Camera coordinates.
  const Matrix3d K1 = K1f.cast<double>();
  const Matrix3d K2 = K2f.cast<double>();
  const Matrix3d K1_inv = K1.inverse();
  const Matrix3d K2_inv = K2.inverse();
  const auto P1 = apply_transform(K1_inv, p1);
  const auto P2 = apply_transform(K2_inv, p2);

  // Normalization transformation.
  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  // Normalized camera coordinates.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);


  // ==========================================================================
  // Generate random samples for RANSAC.
  constexpr auto N = 1000;
  constexpr auto L = EEstimator::num_points;

  // M = list of matches.
  const auto M = to_tensor(matches);
  const auto card_M = M.size(0);

  // S is the list of N groups of L matches.
  const auto S = random_samples(N, L, card_M);
  // Remap a match index to a pair of point indices.
  const auto I = to_point_indices(S, M);

  // Pixel coordinates.
  const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});
  // Camera coordinates.
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});
  // Normalized camera coordinates.
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  auto F_best = FundamentalMatrix{};
  auto E_best = EssentialMatrix{};
  auto num_inliers_best = 0;
  auto subset_best = 0;

  for (auto n = 0; n < N; ++n)
  {
    // Normalized camera coordinates.
    const Matrix<double, 3, L> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Yn = Pn[n][1].colmajor_view().matrix();

    // Some 5-point algorithm.
    auto Es = e_estimator.find_essential_matrices(Xn, Yn);

    // Unnormalize the essential matrices.
    for (auto& E : Es)
    {
      //E.matrix() = E.matrix().normalized();
      E.matrix() = (T2.transpose() * E.matrix() * T1).normalized();
    }

    for (const auto& E: Es)
    {
      const Matrix3d F =
          (K2_inv.transpose() * E.matrix() * K1_inv);  //.normalized();

      const auto num_inliers =
          count_inliers(F, M, p1, p2, epipolar_distance, e_err_thresh);

      if (num_inliers > num_inliers_best)
      {
        num_inliers_best = num_inliers;
        E_best = E;
        F_best = F;
        subset_best = n;
        SARA_CHECK(E_best);
        SARA_CHECK(F_best);
        SARA_CHECK(num_inliers_best);
        SARA_CHECK(subset_best);

        // =====================================================================
        // Visualize.
        //
        // Calculate the projected lines.
        const Matrix<double, 3, L> x = p[n][0].colmajor_view().matrix();
        const Matrix<double, 3, L> y = p[n][1].colmajor_view().matrix();
        //
        Matrix<double, 3, L> proj_x = F_best.matrix() * x;
        proj_x.array().rowwise() /= proj_x.row(2).array();
        //
        Matrix<double, 3, L> proj_y = F_best.matrix().transpose() * y;
        proj_y.array().rowwise() /= proj_y.row(2).array();

        drawer.display_images();

        // Show the inliers.
        for (size_t i = 0; i < matches.size(); ++i)
        {
          const Vector3d x1 = matches[i].x_pos().homogeneous().cast<double>();
          const Vector3d x2 = matches[i].y_pos().homogeneous().cast<double>();

          // inlier predicate.
          if (epipolar_distance(F, x1, x2) > e_err_thresh)
            continue;

          if (i % 20 == 0)
          {
            const Vector3d proj_x1 = F_best.right_epipolar_line(x1);
            const Vector3d proj_x2 = F_best.left_epipolar_line(x2);

            drawer.draw_line_from_eqn(0, proj_x2.cast<float>(), Cyan8, 1);
            drawer.draw_line_from_eqn(1, proj_x1.cast<float>(), Cyan8, 1);
            drawer.draw_match(matches[i], Yellow8, false);
          }
        }

        // Redraw the best group of matches drawn by RANSAC.
        for (auto i = 0; i < L; ++i)
        {
          // Draw the corresponding epipolar lines.
          drawer.draw_line_from_eqn(1, proj_x.col(i).cast<float>(), Magenta8,
                                    1);
          drawer.draw_line_from_eqn(0, proj_y.col(i).cast<float>(), Magenta8,
                                    1);
          drawer.draw_match(matches[S(n, i)], Red8, true);
        }
      }
    }
  }


  SARA_DEBUG << "FINAL RESULT" << endl;

  // Display the result.
  const auto& F = F_best;
  const auto& E = E_best;
  const auto& n = subset_best;

  SARA_CHECK(F);
  SARA_CHECK(E);
  SARA_CHECK(n);
  SARA_CHECK(num_inliers_best);

  drawer.display_images();

  for (size_t i = 0; i < matches.size(); ++i)
  {
    const Vector3d x1 = matches[i].x_pos().homogeneous().cast<double>();
    const Vector3d x2 = matches[i].y_pos().homogeneous().cast<double>();

    // inlier predicate.
    if (epipolar_distance(E, K1_inv * x1, K2_inv * x2) > e_err_thresh)
      continue;

    if (i % 20 == 0)
    {
      const Vector3d proj_x1 = F.right_epipolar_line(x1);
      const Vector3d proj_x2 = F.left_epipolar_line(x2);

      drawer.draw_match(matches[i], Yellow8, false);
      drawer.draw_line_from_eqn(0, proj_x2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_x1.cast<float>(), Cyan8, 1);
    }
  }

  // Extract the points of the best sample drawn by RANSAC.
  const Matrix<double, 3, L> x = p[n][0].colmajor_view().matrix();
  const Matrix<double, 3, L> y = p[n][1].colmajor_view().matrix();

  // Project X to the right image.
  Matrix<double, 3, L> proj_x = F.matrix() * x;
  proj_x.array().rowwise() /= proj_x.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, L> proj_y = F.matrix().transpose() * y;
  proj_y.array().rowwise() /= proj_y.row(2).array();

  for (size_t i = 0; i < L; ++i)
  {
    drawer.draw_match(matches[S(n, i)], Red8, true);
    drawer.draw_line_from_eqn(0, proj_y.col(i).cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(1, proj_x.col(i).cast<float>(), Magenta8, 1);
  }

  get_key();
  close_window();
}
#endif

void estimate_homography(const Image<Rgb8>& image1, const Image<Rgb8>& image2,
                         const KeypointList<OERegion, float>& keys1,
                         const KeypointList<OERegion, float>& keys2,
                         const vector<Match>& matches)
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

  // ==========================================================================
  // Normalize the points.
  const auto to_double = [](const float& src) { return double(src); };
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = extract_centers(f1).cwise_transform(to_double);
  const auto p2 = extract_centers(f2).cwise_transform(to_double);

  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  const auto M = to_tensor(matches);

  // Generate random samples for RANSAC.
  const auto num_samples = 1000;
  const double h_err_thres = 1.;

  auto distance = SymmetricTransferError{};

  const auto [H, num_inliers, sample_best] = ransac(
      M, P1, P2, FourPointAlgorithm{}, distance, num_samples, h_err_thres);

  SARA_CHECK(num_inliers);

  // Display the result.
  drawer.display_images();

  distance = SymmetricTransferError{H};

  for (size_t i = 0; i < matches.size(); ++i)
  {
    const Vector3d X1 = matches[i].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = matches[i].y_pos().cast<double>().homogeneous();

    if (distance(X1, X2) < 1.)
      drawer.draw_match(matches[i], Blue8, false);
  };

  constexpr auto L = FourPointAlgorithm::num_points;
  const auto s_best = sample_best.reshape(Vector2i{1, L});
  const auto I = to_point_indices(s_best, M);
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});

  // Extract the points.
  const Matrix<double, 3, L> X = P[0][0].colmajor_view().matrix();
  const Matrix<double, 3, L> Y = P[0][1].colmajor_view().matrix();

  // Project X to the right image.
  Matrix<double, 3, L> proj_X = H.matrix() * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, L> proj_Y = H.matrix().transpose() * Y;
  proj_Y.array().rowwise() /= proj_Y.row(2).array();

  for (size_t i = 0; i < L; ++i)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[sample_best(i)], Red8, true);

    // Draw the corresponding projected points.
    drawer.draw_point(1, proj_X.col(i).hnormalized().cast<float>(), Magenta8,
                      1);
    drawer.draw_point(0, proj_Y.col(i).hnormalized().cast<float>(), Magenta8,
                      1);
  }

  get_key();
  close_window();
}

void estimate_fundamental_matrix(const Image<Rgb8>& image1,
                                 const Image<Rgb8>& image2,
                                 const KeypointList<OERegion, float>& keys1,
                                 const KeypointList<OERegion, float>& keys2,
                                 const vector<Match>& matches)
{
  // ==========================================================================
  // Setup the visualization.
  const auto scale = 0.25f;
  const auto w = int((image1.width() + image2.width()) * scale);
  const auto h = max(image1.height(), image2.height()) * scale;

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);


  // Transform matches to an array of indices.
  const auto M = to_tensor(matches);

  // ==========================================================================
  // Image coordinates.
  const auto to_double = [](const float& src) { return double(src); };
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto p1 = extract_centers(f1).cwise_transform(to_double);
  const auto p2 = extract_centers(f2).cwise_transform(to_double);

  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  double num_samples = 1000;
  const auto [F, num_inliers, sample_best] = ransac(
      M, P1, P2, FEstimator{}, EpipolarDistance{}, num_samples, f_err_thresh);

  SARA_CHECK(F);
  SARA_CHECK(num_inliers);
  SARA_CHECK(matches.size());

  // Visualize the best sample drawn by RANSAC.
  constexpr auto L = EightPointAlgorithm::num_points;
  const auto s_best = sample_best.reshape(Vector2i{1, L});
  const auto I = to_point_indices(s_best, M);
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});

  // Extract the points.
  const Matrix<double, 3, 8> X = P[0][0].colmajor_view().matrix();
  const Matrix<double, 3, 8> Y = P[0][1].colmajor_view().matrix();

  // Project X to the right image.
  Matrix<double, 3, 8> proj_X = F.matrix() * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, 8> proj_Y = F.matrix().transpose() * Y;
  proj_Y.array().rowwise() /= proj_Y.row(2).array();

  drawer.display_images();

  for (size_t i = 0; i < 8; ++i)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[sample_best(i)], Red8, true);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_Y.col(i).cast<float>(), Magenta8, 1);
  }
  get_key();

  for (size_t i = 0; i < matches.size(); ++i)
  {
    const Vector3d X1 = matches[i].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = matches[i].y_pos().cast<double>().homogeneous();

    if (epipolar_distance(F, X1, X2) > f_err_thresh)
      continue;

    if (i % 100 == 0)
    {
      drawer.draw_match(matches[i], Blue8, false);

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
    }
  }

  get_key();
  close_window();
}

void estimate_essential_matrix(const Image<Rgb8>& image1,
                               const Image<Rgb8>& image2,
                               const Matrix3f& K1f,
                               const Matrix3f& K2f,
                               const KeypointList<OERegion, float>& keys1,
                               const KeypointList<OERegion, float>& keys2,
                               const vector<Match>& matches)
{
  // ==========================================================================
  // Setup the visualization.
  const auto scale = 0.25f;
  const auto w = int((image1.width() + image2.width()) * scale);
  const auto h = max(image1.height(), image2.height()) * scale;

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);
  drawer.display_images();


  // ==========================================================================
  // Image coordinates.
  const auto to_double = [](const float& src) { return double(src); };
  const auto& f1 = features(keys1);
  const auto& f2 = features(keys2);
  const auto u1 = homogeneous(extract_centers(f1).cwise_transform(to_double));
  const auto u2 = homogeneous(extract_centers(f2).cwise_transform(to_double));

  // Camera coordinates.
  const Matrix3d K1 = K1f.cast<double>();
  const Matrix3d K2 = K2f.cast<double>();
  const Matrix3d K1_inv = K1.cast<double>().inverse();
  const Matrix3d K2_inv = K2.cast<double>().inverse();
  const auto u1n = apply_transform(K1_inv, u1);
  const auto u2n = apply_transform(K2_inv, u2);

  const auto M = to_tensor(matches);

  double num_samples = 100;
  auto distance = EpipolarDistance{};

  auto [E, num_inliers, sample_best] =
      ransac(M, u1n, u2n, e_estimator, distance, num_samples, e_err_thresh);

  E.matrix() = E.matrix().normalized();

  SARA_CHECK(E);
  SARA_CHECK(num_inliers);
  SARA_CHECK(matches.size());

  auto F = FundamentalMatrix{};
  F.matrix() = (K2_inv.transpose() * E.matrix() * K1_inv);

  // Visualize the best sample drawn by RANSAC.
  constexpr auto L = EEstimator::num_points;
  const auto s_best = sample_best.reshape(Vector2i{1, L});
  const auto I = to_point_indices(s_best, M);
  const auto u_s = to_coordinates(I, u1, u2).transpose({0, 2, 1, 3});
  const auto un_s = to_coordinates(I, u1n, u2n).transpose({0, 2, 1, 3});

  // Extract the points.
  const Matrix<double, 3, L> u1_s = u_s[0][0].colmajor_view().matrix();
  const Matrix<double, 3, L> u2_s = u_s[0][1].colmajor_view().matrix();
  const Matrix<double, 3, L> u1n_s = un_s[0][0].colmajor_view().matrix();
  const Matrix<double, 3, L> u2n_s = un_s[0][1].colmajor_view().matrix();

  // Project X to the right image.
  Matrix<double, 3, L> proj_u1_s = F.matrix() * u1_s;
  proj_u1_s.array().rowwise() /= proj_u1_s.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, L> proj_u2_s = F.matrix().transpose() * u2_s;
  proj_u2_s.array().rowwise() /= proj_u2_s.row(2).array();

  for (size_t s = 0; s < sample_best.size(); ++s)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(matches[sample_best(s)], Yellow8, true);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_u1_s.col(s).cast<float>(), Red8, 1);
    drawer.draw_line_from_eqn(0, proj_u2_s.col(s).cast<float>(), Red8, 1);
  }

  // Draw the inliers.
  for (size_t i = 0; i < matches.size(); ++i)
  {
    const Vector3d u1_i = matches[i].x_pos().cast<double>().homogeneous();
    const Vector3d u2_i = matches[i].y_pos().cast<double>().homogeneous();

    // inlier predicate.
    if (epipolar_distance(F, u1_i, u2_i) > e_err_thresh)
      continue;

    if (i % 20 == 0)
    {
      const auto proj_u1_i = F.right_epipolar_line(u1_i);
      const auto proj_u2_i = F.left_epipolar_line(u2_i);

      drawer.draw_match(matches[i], Blue8, false);

      drawer.draw_line_from_eqn(0, proj_u2_i.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_u1_i.cast<float>(), Cyan8, 1);
    }
  }


  // Check the cheirality filter.
  const auto candidate_motions = extract_relative_motion_horn(E);

  auto geometries = std::vector<TwoViewGeometry>{};
  std::transform(std::begin(candidate_motions), std::end(candidate_motions),
                 std::back_inserter(geometries), [&](const Motion& m) {
                   return two_view_geometry(m, u1n_s, u2n_s);
                 });
  // for (const auto& g: geometries)
  // {
  //   SARA_DEBUG << "Triangulated points" << std::endl;
  //   const Matrix34d C1 = g.C1;
  //   const Matrix34d C2 = g.C2;
  //   const MatrixXd C1X = C1 * g.X;
  //   const MatrixXd C2X = C2 * g.X;
  //   SARA_DEBUG << "C1 * X =\n" << C1X << std::endl;
  //   SARA_DEBUG << "C2 * X =\n" << C2X << std::endl;

  //   SARA_DEBUG << "(C1 * X).hnormalized() =\n"
  //              << C1X.colwise().hnormalized() << std::endl;
  //   SARA_DEBUG << "u1n_s =\n"
  //              << u1n_s.colwise().hnormalized() << std::endl;

  //   SARA_DEBUG << "(C2 * X).hnormalized() =\n"
  //              << C2X.colwise().hnormalized() << std::endl;
  //   SARA_DEBUG << "u2n_s.hnormalized() =\n"
  //              << u2n_s.colwise().hnormalized() << std::endl;
  //   std::cout << std::endl;
  // }

  // Find the best geometry, i.e., the one with the high cheirality degree.
  const auto best_geom =
      std::max_element(std::begin(geometries), std::end(geometries),
                       [](const auto& g1, const auto& g2) {
                         return g1.cheirality.count() < g2.cheirality.count();
                       });


  const auto cheiral_degree = best_geom->cheirality.count();
  if (cheiral_degree == 0)
    throw std::runtime_error{"The cheirality degree can't be zero!"};

  {
    const Matrix34d P1 = best_geom->C1;
    const Matrix34d P2 = best_geom->C2;

    SARA_DEBUG << "Check cheirality..." << std::endl;
    SARA_DEBUG << "P1 =\n" << P1 << std::endl;
    SARA_DEBUG << "P2 =\n" << P2 << std::endl;
    SARA_DEBUG << "P1 * X =\n" << P1 * best_geom->X << std::endl;
    SARA_DEBUG << "P2 * X =\n" << P2 * best_geom->X << std::endl;

    const auto card_M = M.size(0);
    const auto mindices = range(card_M);
    auto coords_matched = Tensor_<double, 3>{{2, card_M, 3}};
    auto u1n_matched_mat = coords_matched[0].colmajor_view().matrix();
    auto u2n_matched_mat = coords_matched[1].colmajor_view().matrix();
    {
      const auto u1n_mat = u1n.colmajor_view().matrix();
      const auto u2n_mat = u2n.colmajor_view().matrix();
      std::for_each(std::begin(mindices), std::end(mindices), [&](int m) {
        u1n_matched_mat.col(m) = u1n_mat.col(M(m, 0));
        u2n_matched_mat.col(m) = u2n_mat.col(M(m, 1));
      });
    }
    //SARA_DEBUG << "u1n_matched_mat =\n" << u1n_matched_mat.leftCols(10) << std::endl;
    //SARA_DEBUG << "u2n_matched_mat =\n" << u2n_matched_mat.leftCols(10) << std::endl;

    SARA_DEBUG << "Checking epipolar consistency and cheirality by "
                  "triangulating all points..."
               << std::endl;
    auto inlier_predicate = CheiralAndEpipolarConsistency{};
    inlier_predicate.err_threshold = e_err_thresh;
    inlier_predicate.set_model(*best_geom);
    const auto inliers = inlier_predicate(u1n_matched_mat, u2n_matched_mat);
    SARA_CHECK(inliers.count());

    // GeRt the complete geometry.
    auto complete_cheiral_geom = inlier_predicate.geometry;

    // Get the left and right cameras.
    auto cameras = Tensor_<PinholeCamera, 1>{2};
    cameras(0) = complete_cheiral_geom.C1;
    cameras(1) = complete_cheiral_geom.C2;

    // Get the cheiral 3D points.
    auto X = std::vector<Vector3d>{};
    auto colors = std::vector<Rgb8>{};
    X.reserve(M.size(0));
    colors.reserve(M.size(0));

    const auto I1d = image1.convert<Rgb64f>();
    const auto I2d = image2.convert<Rgb64f>();

    std::for_each(std::begin(mindices), std::end(mindices), [&](int m) {
      const bool inlierm = inliers(m);
      const Vector3d Xm = complete_cheiral_geom.X.col(m).hnormalized();
      const Vector2d um = (K1 * u1n_matched_mat.col(m)).hnormalized();
      const Rgb8 rgbm = (interpolate(I1d, um) * 255).cast<unsigned char>();

      if (inlierm &&                         //
          Xm.cwiseAbs().minCoeff() > 1e-3 &&  //
          Xm.cwiseAbs().maxCoeff() < 1e+2)
      {
        X.push_back(Xm);
        colors.push_back(rgbm);
        //SARA_DEBUG << "Adding 3D point: " << Xm.transpose() << std::endl;
        //SARA_CHECK(inlierm);
        //SARA_CHECK(Xm.cwiseAbs().minCoeff());
        //SARA_CHECK(Xm.cwiseAbs().maxCoeff());
      }
    });

    auto X_tensor = TensorView_<double, 2>{reinterpret_cast<double*>(X.data()),
                                          {int(X.size()), 3}};

    SARA_DEBUG << "3D points =\n" << X_tensor.matrix().topRows(10) << std::endl;
    SARA_DEBUG << "Number of 3D valid points = " << X_tensor.size(0)
               << std::endl;
    SARA_DEBUG << "min coeff = " << X_tensor.matrix().minCoeff() << std::endl;
    SARA_DEBUG << "max coeff = " << X_tensor.matrix().maxCoeff() << std::endl;

    auto geom_h5_file =
        H5File{"/Users/david/Desktop/geometry.h5", H5F_ACC_TRUNC};
    geom_h5_file.write_dataset("cameras", cameras, true);
    geom_h5_file.write_dataset("points", X_tensor, true);

    {
      std::filebuf fb;
      fb.open("/Users/david/Desktop/geometry.ply", std::ios::out);
      std::ostream ostr(&fb);
      if (ostr.fail())
        throw std::runtime_error{"Error: failed to create PLY!"};

      tinyply::PlyFile geom_ply_file;
      geom_ply_file.add_properties_to_element(
          "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT64, X_tensor.size(0),
          reinterpret_cast<std::uint8_t*>(X_tensor.data()),
          tinyply::Type::INVALID, 0);
      geom_ply_file.add_properties_to_element(
          "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, X_tensor.size(0),
          reinterpret_cast<std::uint8_t*>(colors.data()),
          tinyply::Type::INVALID, 0);

      geom_ply_file.write(ostr, false);
    }
  }

  get_key();
  close_window();
}


GRAPHICS_MAIN()
{
  // Load images.
  print_stage("Loading images");

  const auto image1 = imread<Rgb8>(data_dir + "/" + file1);
  const auto image2 = imread<Rgb8>(data_dir + "/" + file2);

  const auto K1 = read_internal_camera_parameters(data_dir + "/" + "0000.png.K");
  const auto K2 = read_internal_camera_parameters(data_dir + "/" + "0001.png.K");

  auto keys1 = KeypointList<OERegion, float>{};
  auto keys2 = KeypointList<OERegion, float>{};
  get_keypoints(keys1, keys2);

  const auto matches = compute_matches(keys1, keys2);

#ifdef DEBUG_ME
  estimate_homography_old(image1, image2, keys1, keys2, matches);
  estimate_fundamental_matrix_old(image1, image2, keys1, keys2, matches);
  estimate_essential_matrix_old(image1, image2, K1, K2, keys1, keys2, matches);
#endif

  // estimate_homography(image1, image2, keys1, keys2, matches);
  // estimate_fundamental_matrix(image1, image2, keys1, keys2, matches);
  estimate_essential_matrix(image1, image2, K1, K2, keys1, keys2, matches);

  return 0;
}
