// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#define BOOST_TEST_MODULE "MultiViewGeometry/Essential Matrix"

#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <boost/test/unit_test.hpp>

#include <iomanip>
#include <sstream>


using namespace std;
using namespace DO::Sara;


template <typename T>
void print_3d_array(const TensorView_<T, 3>& x)
{
  const auto max = x.flat_array().abs().maxCoeff();
  std::stringstream ss;
  ss << max;
  const auto pad_size = ss.str().size();


  cout << "[";
  for (auto i = 0; i < x.size(0); ++i)
  {
    cout << "[";
    for (auto j = 0; j < x.size(1); ++j)
    {
      cout << "[";
      for (auto k = 0; k < x.size(2); ++k)
      {
        cout << std::setw(pad_size) << x(i,j,k);
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


// NumPy-like interface for Tensors
auto range(int n) -> Tensor_<int, 1>
{
  auto indices = Tensor_<int, 1>{n};
  std::iota(indices.begin(), indices.end(), 0);
  return indices;
}

auto random_shuffle(const Tensor_<int, 1>& x) -> Tensor_<int, 1>
{
  Tensor_<int, 1> x_shuffled = x;
  std::random_shuffle(x_shuffled.begin(), x_shuffled.end());
  return x_shuffled;
}

auto random_samples(int num_samples, int sample_size, int num_data_points)
    -> Tensor_<int, 2>
{
  auto indices = range(num_data_points);

  auto samples = Tensor_<int, 2>{sample_size, num_samples};
  for (int i = 0; i < sample_size; ++i)
    samples[i].flat_array() =
        random_shuffle(indices).flat_array().head(num_samples);

  samples = samples.transpose({1, 0});

  return samples;
}


// Data transformations.
auto extract_centers(const std::vector<OERegion>& features) -> Tensor_<float, 2>
{
  auto centers = Tensor_<float, 2>{int(features.size()), 2};
  auto mat = centers.matrix();

  for (auto i = 0; i < centers.size(0); ++i)
    mat.row(i) = features[i].center().transpose();

  return centers;
}

auto homogeneous(const TensorView_<float, 2>& x)
  -> Tensor_<float, 2>
{
  auto X = Tensor_<float, 2>(x.size(0), x.size(1) + 1);
  X.matrix().leftCols(x.size(1)) = x.matrix();
  X.matrix().col(x.size(1)).setOnes();
  return X;
}

auto to_point_indices(const TensorView_<int, 2>& samples, const TensorView_<int, 2>& matches)
  -> Tensor_<int, 3>
{
  const auto num_samples = samples.size(0);
  const auto sample_size = samples.size(1);

  auto point_indices = Tensor_<int, 3>{num_samples, sample_size, 2};
  for (auto s = 0; s < num_samples; ++s)
    for (auto m = 0; m < sample_size; ++m)
      point_indices[s][m].flat_array() = matches[samples(s, m)].flat_array();

  return point_indices;
}

auto to_coordinates(const Tensor_<int, 3>& point_indices,
                    const Tensor_<float, 2>& p1,
                    const Tensor_<float, 2>& p2)
  -> Tensor_<float, 4>
{
  const auto num_samples = point_indices.size(0);
  const auto sample_size = point_indices.size(1);
  const auto num_points = 2;
  const auto coords_dim = p1.size(1);

  auto p =
      Tensor_<float, 4>{{num_samples, sample_size, num_points, coords_dim}};

  for (auto s = 0; s < num_samples; ++s)
    for (auto m = 0; m < sample_size; ++m)
    {
      auto p1_idx = point_indices(s, m, 0);
      auto p2_idx = point_indices(s, m, 1);

      p[s][m][0].flat_array() = p1[p1_idx].flat_array();
      p[s][m][1].flat_array() = p2[p2_idx].flat_array();
    }

  return p;
}


// Point transformations.
auto compute_normalizer(const Tensor_<float, 2>& X) -> Matrix3f
{
  const RowVector3f min = X.matrix().colwise().minCoeff();
  const RowVector3f max = X.matrix().colwise().maxCoeff();

  const Matrix2f scale = (max - min).cwiseInverse().head(2).asDiagonal();

  Matrix3f T = Matrix3f::Zero();
  T.topLeftCorner<2, 2>() = scale;
  T.col(2) << -min.cwiseQuotient(max - min).transpose().head(2), 1.f;

  return T;
}

auto apply_transform(const Matrix3f& T, const Tensor_<float, 2>& X)
  -> Tensor_<float, 2>
{
  auto TX = Tensor_<float, 2>{X.sizes()};
  auto TX_ = TX.colmajor_view().matrix();
  TX_ = T * X.colmajor_view().matrix();
  TX_.array().rowwise() /= TX_.array().row(2);
  return TX;
}


BOOST_AUTO_TEST_SUITE(TestMultiViewGeometry)

BOOST_AUTO_TEST_CASE(test_range)
{
  auto a = range(3);
  BOOST_CHECK(vec(a) == Vector3i(0, 1, 2));
}

BOOST_AUTO_TEST_CASE(test_random_shuffle)
{
  auto a = range(4);
  a = random_shuffle(a);
  BOOST_CHECK(vec(a) != Vector4i(0, 1, 2, 3));
}

BOOST_AUTO_TEST_CASE(test_random_samples)
{
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 5;
  constexpr auto num_data_points = 10;
  auto samples = random_samples(num_samples, sample_size, num_data_points);

  BOOST_CHECK_EQUAL(samples.size(0), num_samples);
  BOOST_CHECK_EQUAL(samples.size(1), sample_size);
  BOOST_CHECK(samples.matrix().minCoeff() >=  0);
  BOOST_CHECK(samples.matrix().maxCoeff() <  10);
}


BOOST_AUTO_TEST_CASE(test_extract_centers)
{
  auto features = std::vector<OERegion>{{Point2f::Ones() * 0, 1.f},
                                        {Point2f::Ones() * 1, 1.f},
                                        {Point2f::Ones() * 2, 1.f}};

  const auto x = extract_centers(features);
  auto expected_x = Tensor_<float, 2>{x.sizes()};
  expected_x.matrix() <<
    0, 0,
    1, 1,
    2, 2;

  BOOST_CHECK(x.matrix() == expected_x.matrix());

  const auto X = homogeneous(x);
  auto expected_X = Tensor_<float, 2>{X.sizes()};
  expected_X.matrix() <<
    0, 0, 1,
    1, 1, 1,
    2, 2, 1;
  BOOST_CHECK(X.matrix() == expected_X.matrix());
}

BOOST_AUTO_TEST_CASE(test_to_point_indices)
{
  constexpr auto num_matches = 5;
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 4;

  auto matches = Tensor_<int, 2>{num_matches, 2};
  matches.matrix() <<
    0, 0,
    1, 1,
    2, 2,
    3, 3,
    4, 0;

  auto samples = Tensor_<int, 2>{num_samples, sample_size};
  samples.matrix() <<
    0, 1, 2, 3,
    4, 2, 3, 1;

  auto point_indices = to_point_indices(samples, matches);

  auto expected_point_indices = Tensor_<int, 3>{num_samples, sample_size, 2};
  expected_point_indices[0].matrix() <<
    0, 0,
    1, 1,
    2, 2,
    3, 3;
  expected_point_indices[1].matrix() <<
    4, 0,
    2, 2,
    3, 3,
    1, 1;

  BOOST_CHECK(vec(point_indices) == vec(expected_point_indices));
}

BOOST_AUTO_TEST_CASE(test_to_coordinates)
{
  constexpr auto num_matches = 5;
  constexpr auto num_samples = 2;
  constexpr auto sample_size = 4;

  const auto features1 = std::vector<OERegion>{{Point2f::Ones() * 0, 1.f},
                                               {Point2f::Ones() * 1, 1.f},
                                               {Point2f::Ones() * 2, 1.f}};
  const auto features2 = std::vector<OERegion>{{Point2f::Ones() * 1, 1.f},
                                               {Point2f::Ones() * 2, 1.f},
                                               {Point2f::Ones() * 3, 1.f}};

  const auto points1 = extract_centers(features1);
  const auto points2 = extract_centers(features2);

  auto matches = Tensor_<int, 2>{num_matches, 2};
  matches.matrix() <<
    0, 0,
    1, 1,
    2, 2,
    0, 1,
    1, 2;

  auto samples = Tensor_<int, 2>{num_samples, sample_size};
  samples.matrix() <<
    0, 1, 2, 3,
    1, 2, 3, 4;

  const auto point_indices = to_point_indices(samples, matches);
  const auto coords = to_coordinates(point_indices, points1, points2);

  //                                        N            K            P  C
  auto expected_coords = Tensor_<float, 4>{{num_samples, sample_size, 2, 2}};
  expected_coords[0].flat_array() <<
    0.f, 0.f, 1.f, 1.f,
    1.f, 1.f, 2.f, 2.f,
    2.f, 2.f, 3.f, 3.f,
    0.f, 0.f, 2.f, 2.f;

  expected_coords[1].flat_array() <<
    1.f, 1.f, 2.f, 2.f,
    2.f, 2.f, 3.f, 3.f,
    0.f, 0.f, 2.f, 2.f,
    1.f, 1.f, 3.f, 3.f;

  BOOST_CHECK(vec(expected_coords) == vec(coords));
  BOOST_CHECK(expected_coords.sizes() == coords.sizes());


  const auto coords_t = coords.transpose({0, 2, 1, 3});
  const auto sample1 = coords_t[0];

  auto expected_sample1 = Tensor_<float, 3>{sample1.sizes()};
  expected_sample1.flat_array() <<
    // P1
    0.f, 0.f,
    1.f, 1.f,
    2.f, 2.f,
    0.f, 0.f,
    // P2
    1.f, 1.f,
    2.f, 2.f,
    3.f, 3.f,
    2.f, 2.f;

  //print_3d_array(expected_sample1);
  //print_3d_array(sample1);
  BOOST_CHECK(vec(expected_sample1) == vec(sample1));
}

BOOST_AUTO_TEST_CASE(test_compute_normalizer)
{
  auto X = Tensor_<float, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  auto T = compute_normalizer(X);

  Matrix3f expected_T;
  expected_T <<
    0.5, 0.0, -0.5,
    0.0, 0.5, -0.5,
    0.0, 0.0,  1.0;

  BOOST_CHECK((T - expected_T).norm() < 1e-12);
}

BOOST_AUTO_TEST_CASE(test_apply_transform)
{
  auto X = Tensor_<float, 2>{3, 3};
  X.matrix() <<
    1, 1, 1,
    2, 2, 1,
    3, 3, 1;

  auto T = compute_normalizer(X);

  auto TX = apply_transform(T, X);
}

BOOST_AUTO_TEST_SUITE_END()
