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
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>


using namespace std;
using namespace DO::Sara;


void load(Image<Rgb8>& image1, Image<Rgb8>& image2,
          Set<OERegion, RealDescriptor>& keys1,
          Set<OERegion, RealDescriptor>& keys2,  //
          vector<Match>& matches)
{
  cout << "Loading images" << endl;

  auto data_dir = std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto file1 = "0000.png";
  auto file2 = "0001.png";

  image1 = imread<Rgb8>(data_dir + "/" + file1);
  image2 = imread<Rgb8>(data_dir + "/" + file2);

#ifdef COMPUTE_KEYPOINTS
  cout << "Computing/Reading keypoints" << endl;
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

  // Compute/read matches
  cout << "Computing Matches" << endl;
  AnnMatcher matcher{keys1, keys2, 1.0f};
  matches = matcher.compute_matches();
  cout << matches.size() << " matches" << endl;

  // Debug this.
  //write_matches(matches, data_dir + "/" + "0000_0001.match");
}

auto range(int n) -> VectorXi
{
  auto indices = VectorXi(n);
  std::iota(indices.data(), indices.data() + indices.size(), 0);
  return indices;
}

auto random_shuffle(const VectorXi& x) -> VectorXi
{
  VectorXi x_shuffled = x;
  std::random_shuffle(x_shuffled.data(), x_shuffled.data() + x_shuffled.size());
  return x_shuffled;
}

auto random_samples(int num_points, int sample_size, int num_samples)
    -> MatrixXi
{
  auto indices = range(num_points);

  MatrixXi samples(num_samples, sample_size);
  for (int i = 0; i < sample_size; ++i)
    samples.col(i) = random_shuffle(indices).head(num_samples);

  samples.transposeInPlace();

  return samples;
}

auto to_point_indices(const MatrixXi& samples, const vector<Match>& matches)
  -> Tensor_<int, 3>
{
  auto num_samples = samples.cols();
  auto sample_size = 5;

  Tensor_<int, 3> pts_indices(num_samples, sample_size, 2);
  for (auto s = 0; s < num_samples; ++s)
    for (auto m = 0; m < sample_size; ++m)
      for (auto p = 0; p < 2; ++p)
      {
        const auto match_index = samples(m, s);
        const auto& match = matches[match_index];
        auto p1 = match.x_index();
        auto p2 = match.y_index();
        pts_indices(s, m, 0) = p1;
        pts_indices(s, m, 1) = p2;
      }

  return pts_indices;
}

auto to_coordinates(const Tensor_<int, 3>& samples,
                    const Set<OERegion, RealDescriptor>& keys1,
                    const Set<OERegion, RealDescriptor>& keys2)
  -> std::pair<Tensor_<int, 3>, Tensor_<int, 3>>
{
  auto num_samples = samples.size(0);
  auto sample_size = samples.size(1);

  Tensor_<int, 3> p1(num_samples, sample_size, 2);
  Tensor_<int, 3> p2(num_samples, sample_size, 2);

  for (auto s = 0; s < num_samples; ++s)
    for (auto m = 0; m < sample_size; ++m)
    {
      auto p1_idx = samples(s, m, 0);
      auto p2_idx = samples(s, m, 1);

      p1(s, m, 0) = keys1.f(p1_idx).x();
      p1(s, m, 1) = keys1.f(p1_idx).y();

      p2(s, m, 0) = keys2.f(p2_idx).x();
      p2(s, m, 1) = keys2.f(p2_idx).y();
    }

  return make_pair(p1, p2);
}

auto extract_centers(const Set<OERegion, RealDescriptor>& keys) -> MatrixXf
{
  MatrixXf centers(3, keys.features.size());

  for (auto i = 0; i < keys.size(); ++i)
    centers.col(i) << keys.f(i).center(), 1.f;

  return centers;
}

auto transform(const MatrixXd& K, const MatrixXd& X)
  -> MatrixXd
{
  MatrixXd KX = K*X;
  KX.rowwise() /= KX.row(2);
  return KX;
}

void standardize(const MatrixXd& X)
{
}


GRAPHICS_MAIN()
{
  // Load images.
  auto image1 = Image<Rgb8>{};
  auto image2 = Image<Rgb8>{};
  auto keys1 = Set<OERegion, RealDescriptor>{};
  auto keys2 = Set<OERegion, RealDescriptor>{};
  auto matches = vector<Match>{};
  load(image1, image2, keys1, keys2, matches);

  auto scale = 0.25f;
  auto w = int((image1.width() + image2.width()) * scale);
  auto h = max(image1.height(), image2.height()) * scale;
  auto off = Point2f{float(image1.width()), 0.f};

  create_window(w, h);
  set_antialiasing();

  for (size_t i = 0; i < matches.size(); ++i)
  {
    draw_image_pair(image1, image2, off, scale);
    draw_match(matches[i], Red8, off, scale);
    cout << matches[i] << endl;
    get_key();
  }

  return 0;
}
