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


void estimate_essential_matrix(const Set<OERegion, RealDescriptor>& keys1,
                               const Set<OERegion, RealDescriptor>& keys2,
                               const vector<Match>& matches)
{
  // Convert.
  auto to_tensor = [](const vector<Match>& matches) -> Tensor_<int, 2> {
    auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
    for (auto i = 0u; i < matches.size(); ++i)
      match_tensor[i].flat_array() << matches[i].x_index(),
          matches[i].y_index();
    return match_tensor;
  };

  const auto p1 = extract_centers(keys1.features);
  const auto p2 = extract_centers(keys2.features);

  auto P1 = homogeneous(p1);
  auto P2 = homogeneous(p2);

  //const auto K1 = read_internal_camera_parameters(data_dir + "/" + "0000.png.K");
  //const auto K2 = read_internal_camera_parameters(data_dir + "/" + "0001.png.K");

  //P1 = apply_transform(K1, P1);
  //P2 = apply_transform(K2, P2);

  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);

  //print_3d_array(P[0]);

  constexpr auto N = 1000;
  constexpr auto L = 8;

  const auto M = to_tensor(matches);
  const auto S = random_samples(N, L, M.size(0));
  const auto I = to_point_indices(S, M);
  const auto P = to_coordinates(I, p1, p2);

  std::cout << I[0].matrix() << std::endl;
}

void estimate_fundamental_matrix(const Image<Rgb8>& image1,
                                 const Image<Rgb8>& image2,
                                 const Set<OERegion, RealDescriptor>& keys1,
                                 const Set<OERegion, RealDescriptor>& keys2,
                                 const vector<Match>& matches)
{
  // Convert.
  auto to_tensor = [](const vector<Match>& matches) -> Tensor_<int, 2> {
    auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
    for (auto i = 0u; i < matches.size(); ++i)
      match_tensor[i].flat_array() << matches[i].x_index(),
          matches[i].y_index();
    return match_tensor;
  };

  const auto p1 = extract_centers(keys1.features);
  const auto p2 = extract_centers(keys2.features);

  auto P1 = homogeneous(p1);
  auto P2 = homogeneous(p2);

  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  // Normalize the points.
  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);

  constexpr auto N = 1000;
  constexpr auto L = 8;

  const auto M = to_tensor(matches);
  const auto S = random_samples(N, L, M.size(0));
  const auto I = to_point_indices(S, M);
  const auto p = to_coordinates(I, p1, p2).transpose({0, 2, 1, 3});
  const auto P = to_coordinates(I, P1, P2).transpose({0, 2, 1, 3});
  const auto Pn = to_coordinates(I, P1n, P2n).transpose({0, 2, 1, 3});

  const Matrix<float, 2, 8> x = p[0][0].colmajor_view().matrix();
  const Matrix<float, 2, 8> y = p[0][1].colmajor_view().matrix();

  const Matrix<double, 3, 8> X = P[0][0].colmajor_view().matrix().cast<double>();
  const Matrix<double, 3, 8> Y = P[0][1].colmajor_view().matrix().cast<double>();

  const Matrix<double, 3, 8> Xn = Pn[0][0].colmajor_view().matrix().cast<double>();
  const Matrix<double, 3, 8> Yn = Pn[0][1].colmajor_view().matrix().cast<double>();


  auto F = FundamentalMatrix<>{};
  eight_point_fundamental_matrix(Xn, Yn, F);

  // Unnormalize.
  F.matrix() = T1.cast<double>().inverse().transpose() * F.matrix() *
               T2.cast<double>().inverse();

  std::cout << F.matrix() << std::endl;

  // Projected X to the right.
  Matrix<double, 3, 8> proj_X = F.matrix().transpose() * X;
  proj_X.array().rowwise() /= proj_X.row(2).array();
  std::cout << proj_X << std::endl;



  print_stage("Visualizing matches...");
  auto scale = 1.f;
  auto w = int((image1.width() + image2.width()) * scale);
  auto h = max(image1.height(), image2.height()) * scale;

  create_window(w, h);
  set_antialiasing();

  PairWiseDrawer drawer(image1, image2);
  drawer.set_viz_params(1.f, 1.f, PairWiseDrawer::CatH);
  drawer.display_images();

  for (size_t i = 0; i < 8; ++i)
  {
    //drawer.draw_point(0,matches[i], Red8, true);
    drawer.draw_point(0, x.col(i), Red8, 5); 
    drawer.draw_point(1, y.col(i), Red8, 5); 
    drawer.draw_line_from_eqn(1, proj_X.col(i).cast<float>(), Magenta8, 5);
    //cout << matches[i] << endl;
  }

  get_key();
}

void estimate_homography(const Set<OERegion, RealDescriptor>& keys1,
                         const Set<OERegion, RealDescriptor>& keys2,
                         const vector<Match>& matches)
{
  // Convert.
  auto to_tensor = [](const vector<Match>& matches) -> Tensor_<int, 2> {
    auto match_tensor = Tensor_<int, 2>{int(matches.size()), 2};
    for (auto i = 0u; i < matches.size(); ++i)
      match_tensor[i].flat_array() << matches[i].x_index(),
          matches[i].y_index();
    return match_tensor;
  };

  const auto p1 = extract_centers(keys1.features);
  const auto p2 = extract_centers(keys2.features);

  auto P1 = homogeneous(p1);
  auto P2 = homogeneous(p2);

  auto T1 = compute_normalizer(P1);
  auto T2 = compute_normalizer(P2);

  const auto P1n = apply_transform(T1, P1);
  const auto P2n = apply_transform(T2, P2);

  constexpr auto N = 1000;
  constexpr auto L = 4;

  const auto M = to_tensor(matches);
  const auto S = random_samples(N, L, M.size(0));
  const auto I = to_point_indices(S, M);
  const auto P = to_coordinates(I, p1, p2);
}

GRAPHICS_MAIN()
{
  // Load images.
  print_stage("Loading images");

  const auto image1 = imread<Rgb8>(data_dir + "/" + file1);
  const auto image2 = imread<Rgb8>(data_dir + "/" + file2);

  auto keys1 = Set<OERegion, RealDescriptor>{};
  auto keys2 = Set<OERegion, RealDescriptor>{};
  compute_keypoints(keys1, keys2);

  const auto matches = compute_matches(keys1, keys2);

  estimate_fundamental_matrix(image1, image2, keys1, keys2, matches);

  //print_stage("Visualizing matches...");
  //auto scale = 0.25f;
  //auto w = int((image1.width() + image2.width()) * scale);
  //auto h = max(image1.height(), image2.height()) * scale;
  //auto off = Point2f{float(image1.width()), 0.f};

  //create_window(w, h);
  //set_antialiasing();

  //for (size_t i = 0; i < matches.size(); ++i)
  //{
  //  draw_image_pair(image1, image2, off, scale);
  //  draw_match(matches[i], Red8, off, scale);
  //  cout << matches[i] << endl;
  //  get_key();
  //}

  return 0;
}
