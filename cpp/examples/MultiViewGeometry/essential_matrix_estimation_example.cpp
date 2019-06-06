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
    std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
const auto file1 = "0000.png";
const auto file2 = "0001.png";


void print_3d_array(const TensorView_<float, 3>& x)
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


Matrix3f read_internal_camera_parameters(const std::string& filepath)
{
  std::ifstream file{filepath};
  if (!file)
    throw std::runtime_error{"File " + filepath + "does not exist!"};

  Matrix3f K;
  file >> K;

  return K;
}


void compute_keypoints(Set<OERegion, RealDescriptor>& keys1,
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

vector<Match> compute_matches(const Set<OERegion, RealDescriptor>& keys1,
                              const Set<OERegion, RealDescriptor>& keys2)
{
  print_stage("Computing Matches");
  AnnMatcher matcher{keys1, keys2, 0.6f};

  const auto matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;

  return matches;
}


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
    const Matrix<float, 2, 8> x = p[n][0].colmajor_view().matrix();
    const Matrix<float, 2, 8> y = p[n][1].colmajor_view().matrix();

    const Matrix<double, 3, 8> X =
        P[n][0].colmajor_view().matrix().cast<double>();
    const Matrix<double, 3, 8> Y =
        P[n][1].colmajor_view().matrix().cast<double>();

    const Matrix<double, 3, 8> Xn =
        Pn[n][0].colmajor_view().matrix().cast<double>();
    const Matrix<double, 3, 8> Yn =
        Pn[n][1].colmajor_view().matrix().cast<double>();
    //std::cout << Xn << std::endl << std::endl;
    //std::cout << Yn << std::endl;


    // 8-point algorithm
    auto F = FundamentalMatrix<>{};
    eight_point_fundamental_matrix(Xn, Yn, F);

    //std::cout << "Check normalized F..." << std::endl;
    //std::cout << "F = " << std::endl;
    //std::cout << F.matrix() << std::endl;
    //std::cout << "Algebraic errors:" << std::endl;
    //for (int i = 0; i < 8; ++i)
    //  std::cout << Xn.col(i).transpose() * F.matrix() * Yn.col(i) << std::endl;


    // Unnormalize the fundamental matrix.
    F.matrix() = T1.cast<double>().transpose() * F.matrix() * T2.cast<double>();
    //F.matrix() = F.matrix().normalized();

    std::cout << "Check unnormalized F..." << std::endl;
    std::cout << "F = " << std::endl;
    std::cout << F.matrix() << std::endl;
    std::cout << "Algebraic errors:" << std::endl;
    for (int i = 0; i < 8; ++i)
      std::cout << X.col(i).transpose() * F.matrix() * Y.col(i) << std::endl;


    // Projected X to the right.
    Matrix<double, 3, 8> proj_X = F.matrix() * X;
    proj_X.array().rowwise() /= proj_X.row(2).array();

    Matrix<double, 3, 8> proj_Y = F.matrix().transpose() * Y;
    proj_Y.array().rowwise() /= proj_Y.row(2).array();


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


      Vector3d proj_X1 = F.matrix().transpose() * X1;
      proj_X1 /= proj_X1(2);

      Vector3d proj_X2 = F.matrix() * X2;
      proj_X2 /= proj_X2(2);


      if (std::abs(X1.transpose() * F.matrix() * X2) < 1e-2)
      {
        drawer.draw_match(matches[i], Blue8, false);
        //drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
        //drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
      }
    };


    for (size_t i = 0; i < 8; ++i)
    {
      drawer.draw_match(matches[S(n, i)], Red8, true);

      drawer.draw_point(0, x.col(i), Magenta8, 5);
      drawer.draw_point(1, y.col(i), Magenta8, 5);

      drawer.draw_point(0, X.col(i).cast<float>().head(2), Magenta8, 5);
      drawer.draw_point(1, Y.col(i).cast<float>().head(2), Magenta8, 5);

      drawer.draw_line_from_eqn(0, proj_Y.col(i).cast<float>(), Magenta8, 1);
      drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(), Magenta8, 1);
    }

    get_key();
  }
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

  // TODO: by default a feature matcher should just return a tensor. It is
  // more cryptic but more powerful to manipulate data.
  //
  // Convert a set of matches to a tensor.
  auto to_tensor = [](const vector<Match>& matches) -> Tensor_<int, 2> {
    auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
    for (auto i = 0u; i < matches.size(); ++i)
      match_tensor[i].flat_array() << matches[i].x_index(),
          matches[i].y_index();
    return match_tensor;
  };

  // Convert to double.
  Matrix3d K1 = K1f.cast<double>();
  Matrix3d K2 = K2f.cast<double>();

  // Image coordinates.
  const auto to_double = [](const float& src) { return double(src); };
  const auto p1 = extract_centers(keys1.features).cwise_transform(to_double);
  const auto p2 = extract_centers(keys2.features).cwise_transform(to_double);

  // Camera coordinates.
  auto P1 = apply_transform(K1.inverse().eval(), homogeneous(p1));
  auto P2 = apply_transform(K2.inverse().eval(), homogeneous(p2));

  // Normalization transformation.
  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  // Normalized camera coordinates.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);

  //print_3d_array(P[0]);

  // RANSAC draws N groups of L matches.
  constexpr auto N = 1000;
  constexpr auto L = 5;

  // M = list of matches.
  const auto M = to_tensor(matches);
  const auto card_M = M.size(0);

  // The RANSAC procedure.
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

  // Nister 5-point algorithm.
  auto solver = NisterFivePointAlgorithm{};

  for (auto n = 0; n < N; ++n)
  {
    // Pixel coordinates of the 5 matches.
    const Matrix<double, 2, L> x = p[n][0].colmajor_view().matrix();
    const Matrix<double, 2, L> y = p[n][1].colmajor_view().matrix();

    const Matrix<double, 3, L> X = P[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Y = P[n][1].colmajor_view().matrix();

    const Matrix<double, 3, L> Xn = Pn[n][0].colmajor_view().matrix();
    const Matrix<double, 3, L> Yn = Pn[n][1].colmajor_view().matrix();

    // 5-point algorithm.
    const auto Es = solver.find_essential_matrices(Xn, Yn);
    for (const auto& E: Es)
    {
      // Display the result.
      drawer.display_images();

      for (size_t i = 0; i < matches.size(); ++i)
      {
        // Image coordinates.
        Vector3d x1;
        x1.head(2) = matches[i].x_pos().cast<double>();
        x1(2) = 1;

        Vector3d x2;
        x2.head(2) = matches[i].y_pos().cast<double>();
        x2(2) = 1;

        // Normalized camera coordinates
        Vector3d X1n = T1 * K1.inverse() * x1;
        Vector3d X2n = T1 * K1.inverse() * x2;

        //Vector3d proj_X1 = E.transpose() * X1;
        //proj_X1 /= proj_X1(2);

        //Vector3d proj_X2 = E * X2;
        //proj_X2 /= proj_X2(2);


        if (std::abs(X2n.transpose() * E * X1n) < 1e-2)
        {
          drawer.draw_match(matches[i], Blue8, false);
          //drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
          //drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
        }
      };


      for (size_t i = 0; i < L; ++i)
      {
        drawer.draw_match(matches[S(n, i)], Red8, true);

        //drawer.draw_point(0, x.col(i).cast<float>(), Magenta8, 5);
        //drawer.draw_point(1, y.col(i).cast<float>(), Magenta8, 5);

        //drawer.draw_point(0, X.col(i).cast<float>().head(2), Magenta8, 5);
        //drawer.draw_point(1, Y.col(i).cast<float>().head(2), Magenta8, 5);

        // drawer.draw_line_from_eqn(0, proj_Y.col(i).cast<float>(), Magenta8,
        // 1); drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(),
        // Magenta8, 1);
      }

      get_key();
    }
  }
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

  //estimate_homography(image1, image2, keys1, keys2, matches);
  estimate_fundamental_matrix(image1, image2, keys1, keys2, matches);
  //estimate_essential_matrix(image1, image2, K1, K2, keys1, keys2, matches);

  return 0;
}
