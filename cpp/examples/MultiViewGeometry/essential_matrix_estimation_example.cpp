// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "sift.hpp"

#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/FivePointAlgorithms.hpp>


using namespace std;
using namespace DO::Sara;


const auto data_dir =
    std::string{"/Users/David/Desktop/Datasets/sfm/castle_int"};
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

auto to_tensor(const vector<Match>& matches)
{
  auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
  for (auto i = 0u; i < matches.size(); ++i)
    match_tensor[i].flat_array() << matches[i].x_index(), matches[i].y_index();
  return match_tensor;
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
auto compute_keypoints(Set<OERegion, RealDescriptor>& keys1,
                       Set<OERegion, RealDescriptor>& keys2)
{
  print_stage("Computing/Reading keypoints");

#ifdef COMPUTE_KEYPOINTS
  auto sifts1 = compute_sift_keypoints(image1.convert<float>());
  auto sifts2 = compute_sift_keypoints(image2.convert<float>());
  keys1.append(sifts1);
  keys2.append(sifts2);
  cout << "Image 1: " << keys1.size() << " keypoints" << endl;
  cout << "Image 2: " << keys2.size() << " keypoints" << endl;

  write_keypoints(sifts1.features, sifts1.descriptors,
                  data_dir + "/" + "0000.key");
  write_keypoints(sifts2.features, sifts2.descriptors,
                  data_dir + "/" + "0001.key");

#else
  read_keypoints(keys1.features, keys1.descriptors,
                 data_dir + "/" + "0000.key");
  read_keypoints(keys2.features, keys2.descriptors,
                 data_dir + "/" + "0001.key");
#endif
}

// TODO: by default a feature matcher should just return a tensor. It is
// more cryptic but more powerful to manipulate data.
//
// Convert a set of matches to a tensor.
auto compute_matches(const Set<OERegion, RealDescriptor>& keys1,
                     const Set<OERegion, RealDescriptor>& keys2)
{
  print_stage("Computing Matches");
  AnnMatcher matcher{keys1, keys2, 0.6f};

  const auto matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;

  return matches;
}


// =============================================================================
// Multiview geometry estimation.
//
void estimate_homography(const Image<Rgb8>& image1, const Image<Rgb8>& image2,
                         const Set<OERegion, RealDescriptor>& keys1,
                         const Set<OERegion, RealDescriptor>& keys2,
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
  // Generate random samples for RANSAC.
  constexpr auto N = 1000;
  constexpr auto L = 4;
  const auto S = random_samples(N, L, int(matches.size()));
  //auto S = Tensor_<int, 2>{{1, L}};
  //S[0] = range(8);




  // ==========================================================================
  // Normalize the points.
  const auto p1 = extract_centers(keys1.features);
  const auto p2 = extract_centers(keys2.features);

  auto P1 = homogeneous(p1);
  auto P2 = homogeneous(p2);

  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);




  // ==========================================================================
  // Prepare the data for RANSAC.
  auto to_tensor = [](const vector<Match>& matches) -> Tensor_<int, 2> {
    auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
    for (auto i = 0u; i < matches.size(); ++i)
      match_tensor[i].flat_array() << matches[i].x_index(),
          matches[i].y_index();
    return match_tensor;
  };
  const auto M = to_tensor(matches);
  const auto I = to_point_indices(S, M);
  const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});


  for (auto n = 0; n < N; ++n)
  {
    // Extract the point
    const Matrix<float, 2, 4> x = p[n][0].colmajor_view().matrix();
    const Matrix<float, 2, 4> y = p[n][1].colmajor_view().matrix();

    const Matrix<double, 3, 4> X =
        P[n][0].colmajor_view().matrix().cast<double>();
    const Matrix<double, 3, 4> Y =
        P[n][1].colmajor_view().matrix().cast<double>();

    const Matrix<double, 3, 4> Xn =
        Pn[n][0].colmajor_view().matrix().cast<double>();
    const Matrix<double, 3, 4> Yn =
        Pn[n][1].colmajor_view().matrix().cast<double>();


    // 4-point algorithm
    auto H = Matrix3d{};
    four_point_homography(Xn, Yn, H);

    // Unnormalize the fundamental matrix.
    H = T2.cast<double>().inverse() * H * T1.cast<double>();

    std::cout << "Check H..." << std::endl;
    std::cout << H << std::endl;

    MatrixXd HX = H * X;
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
      Vector3d X1;
      X1.head(2) = matches[i].x_pos().cast<double>();
      X1(2) = 1;

      Vector3d X2;
      X2.head(2) = matches[i].y_pos().cast<double>();
      X2(2) = 1;

      Vector3d proj_X1 = H * X1;
      proj_X1 /= proj_X1(2);

      Vector3d proj_X2 = H.inverse() * X2;
      proj_X2 /= proj_X2(2);


      if ((X1 - proj_X2).norm() + (proj_X1 - X2).norm() < 2)
      {
        drawer.draw_match(matches[i], Blue8, false);
        drawer.draw_point(0, proj_X2.head(2).cast<float>(), Cyan8, 5);
        drawer.draw_point(1, proj_X1.head(2).cast<float>(), Cyan8, 5);
      }
    };
    for (size_t i = 0; i < 4; ++i)
    {
      drawer.draw_match(matches[S(n, i)], Red8, true);

      drawer.draw_point(0, x.col(i), Magenta8, 5);
      drawer.draw_point(1, y.col(i), Magenta8, 5);

      drawer.draw_point(0, X.col(i).cast<float>().head(2), Magenta8, 5);
      drawer.draw_point(1, Y.col(i).cast<float>().head(2), Magenta8, 5);

      drawer.draw_point(1, HX.col(i).cast<float>().head(2), Blue8, 5);

      // cout << matches[i] << endl;
    }

    get_key();
  }
}

void estimate_fundamental_matrix(const Image<Rgb8>& image1,
                                 const Image<Rgb8>& image2,
                                 const Set<OERegion, RealDescriptor>& keys1,
                                 const Set<OERegion, RealDescriptor>& keys2,
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
  // Generate random samples for RANSAC.
  constexpr auto N = 1000;
  constexpr auto L = 8;
  const auto S = random_samples(N, L, int(matches.size()));


  // ==========================================================================
  // Image coordinates.
  const auto to_double = [](const float& src) { return double(src); };
  const auto p1 = extract_centers(keys1.features).cwise_transform(to_double);
  const auto p2 = extract_centers(keys2.features).cwise_transform(to_double);

  const auto P1 = homogeneous(p1);
  const auto P2 = homogeneous(p2);

  // Normalization transformation.
  const auto T1 = compute_normalizer(P1);
  const auto T2 = compute_normalizer(P2);

  // Normalized image coordinates.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);


  // ==========================================================================
  // Prepare the data for RANSAC.
  const auto M = to_tensor(matches);
  const auto I = to_point_indices(S, M);
  const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  auto F_best = Matrix<double, 3, 3>{};
  auto num_inliers_best = 0;
  auto subset_best = 0;

  auto algebraic_error = [](const auto& F, const auto& X, const auto& Y) {
    return std::abs(Y.transpose() * F.matrix() * X);
  };

  auto inlier_predicate = [&](const auto& F, const auto& X, const auto& Y) {
    return algebraic_error(F, X, Y) < 1e-2;
  };

  for (auto n = 0; n < N; ++n)
  {
    // Extract the corresponding points.
    const Matrix<double, 3, 8> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, 8> Yn = Pn[n][1].colmajor_view().matrix();

    // Estimate the fundamental matrix.
    auto F = FundamentalMatrix<>{};
    eight_point_fundamental_matrix(Xn, Yn, F);

    // Unnormalize the fundamental matrix.
    F.matrix() = T2.transpose() * F.matrix().normalized() * T1;
    F.matrix() = F.matrix().normalized();

    // Count the inliers.
    auto num_inliers = 0;
    for (size_t i = 0; i < matches.size(); ++i)
    {
      Vector3d X1;
      X1.head(2) = matches[i].x_pos().cast<double>();
      X1(2) = 1;

      Vector3d X2;
      X2.head(2) = matches[i].y_pos().cast<double>();
      X2(2) = 1;

      // inlier predicate.
      if (!inlier_predicate(F, X1, X2))
        continue;

      ++num_inliers;
    }

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

  SARA_CHECK(F);
  SARA_CHECK(n);

  // Extract the points.
  const Matrix<double, 3, 8> X = P[n][0].colmajor_view().matrix();
  const Matrix<double, 3, 8> Y = P[n][1].colmajor_view().matrix();

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

  for (size_t i = 0; i < matches.size(); ++i)
  {
    Vector3d X1;
    X1.head(2) = matches[i].x_pos().cast<double>();
    X1(2) = 1;

    Vector3d X2;
    X2.head(2) = matches[i].y_pos().cast<double>();
    X2(2) = 1;

    // inlier predicate.
    if (!inlier_predicate(F, X1, X2))
      continue;

    drawer.draw_match(matches[i], Blue8, false);

    if (i % 50 == 0)
    {
      Vector3d proj_X1 = F.matrix() * X1;
      proj_X1 /= proj_X1(2);

      Vector3d proj_X2 = F.matrix().transpose() * X2;
      proj_X2 /= proj_X2(2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
    }
  }

  get_key();
}

void estimate_essential_matrix(const Image<Rgb8>& image1,
                               const Image<Rgb8>& image2,
                               const Matrix3f& K1f,
                               const Matrix3f& K2f,
                               const Set<OERegion, RealDescriptor>& keys1,
                               const Set<OERegion, RealDescriptor>& keys2,
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
  const auto p1 = extract_centers(keys1.features).cwise_transform(to_double);
  const auto p2 = extract_centers(keys2.features).cwise_transform(to_double);

  // Camera coordinates.
  const Matrix3d K1 = K1f.cast<double>();
  const Matrix3d K2 = K2f.cast<double>();
  const Matrix3d K1_inv = K1.inverse();
  const Matrix3d K2_inv = K2.inverse();
  const auto P1 = apply_transform(K1_inv, homogeneous(p1));
  const auto P2 = apply_transform(K2_inv, homogeneous(p2));

  // Normalization transformation.
  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  // Normalized camera coordinates.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);


  // ==========================================================================
  // Generate random samples for RANSAC.
  constexpr auto N = 1000;
  constexpr auto L = 5;

  // M = list of matches.
  const auto M = to_tensor(matches);
  const auto card_M = M.size(0);

  // S is the list of N groups of L matches.
  const auto S = random_samples(N, L, card_M);
  // Remap a match index to a pair of point indices.
  const auto I = to_point_indices(S, M);

  // Pixel coordinates.
  const auto p = to_coordinates(I, homogeneous(p1), homogeneous(p2))
                     .transpose({0, 2, 1, 3});
  // Camera coordinates.
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});
  // Normalized camera coordinates.
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  // Nister 5-point algorithm.
  auto solver = NisterFivePointAlgorithm{};

  auto F_best = Matrix<double, 3, 3>{};
  auto E_best = Matrix<double, 3, 3>{};
  auto num_inliers_best = 0;
  auto subset_best = 0;

  auto algebraic_error = [](const auto& E, const auto& X, const auto& Y) {
    return std::abs(Y.transpose() * E * X);
  };

  auto inlier_predicate = [&](const auto& E, const auto& X, const auto& Y) {
    return algebraic_error(E, X, Y) < 1e-3;
  };

  for (auto n = 0; n < N; ++n)
  {
    // Normalized camera coordinates.
    const Matrix<double, 3, L> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Yn = Pn[n][1].colmajor_view().matrix();

    // Nister's 5-point algorithm.
    auto Es = solver.find_essential_matrices(Xn, Yn);

    // Unnormalize the essential matrices.
    for (auto& E : Es)
    {
      E = E.normalized();
      E = (T2.transpose() * E * T1).normalized();
    }

    for (const auto& E: Es)
    {
      const Matrix3d F = (K2_inv.transpose() * E * K1_inv).normalized();

      auto num_inliers = 0;
      for (size_t i = 0; i < matches.size(); ++i)
      {
        // Image coordinates.
        Vector3d x1;
        x1.head(2) = matches[i].x_pos().cast<double>();
        x1(2) = 1;

        Vector3d x2;
        x2.head(2) = matches[i].y_pos().cast<double>();
        x2(2) = 1;

        if (!inlier_predicate(F, x1, x2))
          continue;

        ++num_inliers;
      }

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
        Matrix<double, 3, L> proj_x = F * x;
        proj_x.array().rowwise() /= proj_x.row(2).array();
        //
        Matrix<double, 3, L> proj_y = F.transpose() * y;
        proj_y.array().rowwise() /= proj_y.row(2).array();

        drawer.display_images();

        // Show the inliers.
        for (size_t i = 0; i < matches.size(); ++i)
        {
          Vector3d x1;
          x1.head(2) = matches[i].x_pos().cast<double>();
          x1(2) = 1;

          Vector3d x2;
          x2.head(2) = matches[i].y_pos().cast<double>();
          x2(2) = 1;

          // inlier predicate.
          if (!inlier_predicate(F, x1, x2))
            continue;

          if (i % 20 == 0)
          {
            Vector3d proj_x1 = F.matrix() * x1;
            proj_x1 /= proj_x1(2);

            Vector3d proj_x2 = F.matrix().transpose() * x2;
            proj_x2 /= proj_x2(2);

            drawer.draw_line_from_eqn(0, proj_x2.cast<float>(), Cyan8, 1);
            drawer.draw_line_from_eqn(1, proj_x1.cast<float>(), Cyan8, 1);
            drawer.draw_match(matches[i], Yellow8, false);
          }
          else
            drawer.draw_match(matches[i], Blue8, false);

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

  // Extract the points.
  const Matrix<double, 3, L> x = p[n][0].colmajor_view().matrix();
  const Matrix<double, 3, L> y = p[n][1].colmajor_view().matrix();

  // Project X to the right image.
  Matrix<double, 3, L> proj_x = F.matrix() * x;
  proj_x.array().rowwise() /= proj_x.row(2).array();

  // Project Y to the left image.
  Matrix<double, 3, L> proj_y = F.matrix().transpose() * y;
  proj_y.array().rowwise() /= proj_y.row(2).array();

  for (size_t i = 0; i < matches.size(); ++i)
  {
    Vector3d x1;
    x1.head(2) = matches[i].x_pos().cast<double>();
    x1(2) = 1;

    Vector3d x2;
    x2.head(2) = matches[i].y_pos().cast<double>();
    x2(2) = 1;

    // inlier predicate.
    if (!inlier_predicate(E, K1_inv * x1, K2_inv * x2))
      continue;

    if (i % 50 == 0)
    {
      drawer.draw_match(matches[i], Yellow8, false);

      Vector3d proj_x1 = F.matrix() * x1;
      proj_x1 /= proj_x1(2);

      Vector3d proj_x2 = F.matrix().transpose() * x2;
      proj_x2 /= proj_x2(2);

      drawer.draw_line_from_eqn(0, proj_x2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_x1.cast<float>(), Cyan8, 1);
    }
    else
      drawer.draw_match(matches[i], Blue8, false);

  }

  for (size_t i = 0; i < L; ++i)
  {
    Vector3d x1;
    x1.head(2) = matches[i].x_pos().cast<double>();
    x1(2) = 1;

    Vector3d x2;
    x2.head(2) = matches[i].y_pos().cast<double>();
    x2(2) = 1;

    Vector3d proj_x1 = F.matrix() * x1;
    proj_x1 /= proj_x1(2);

    Vector3d proj_x2 = F.matrix().transpose() * x2;
    proj_x2 /= proj_x2(2);

    drawer.draw_match(matches[S(n, i)], Red8, true);
    drawer.draw_line_from_eqn(0, proj_x2.cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(1, proj_x1.cast<float>(), Magenta8, 1);
  }

  get_key();
}


GRAPHICS_MAIN()
{
  // Load images.
  print_stage("Loading images");

  const auto image1 = imread<Rgb8>(data_dir + "/" + file1);
  const auto image2 = imread<Rgb8>(data_dir + "/" + file2);

  const auto K1 = read_internal_camera_parameters(data_dir + "/" + "0000.png.K");
  const auto K2 = read_internal_camera_parameters(data_dir + "/" + "0001.png.K");

  auto keys1 = Set<OERegion, RealDescriptor>{};
  auto keys2 = Set<OERegion, RealDescriptor>{};
  compute_keypoints(keys1, keys2);

  const auto matches = compute_matches(keys1, keys2);

  // estimate_homography(image1, image2, keys1, keys2, matches);
  // estimate_fundamental_matrix(image1, image2, keys1, keys2, matches);
  estimate_essential_matrix(image1, image2, K1, K2, keys1, keys2, matches);

  return 0;
}
