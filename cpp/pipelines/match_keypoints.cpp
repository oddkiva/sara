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

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;


namespace DO::Sara {

auto match(const KeypointList<OERegion, float>& keys1,
           const KeypointList<OERegion, float>& keys2)
    -> std::vector<Match>
{
  AnnMatcher matcher{keys1, keys2, 1.0f};
  return matcher.compute_matches();
}

auto estimate_fundamental_matrix(const std::vector<Match>& Mij,
                                 const KeypointList<OERegion, float>& ki,
                                 const KeypointList<OERegion, float>& kj)
{
  const auto to_double = [](const float& src) { return double(src); };
  const auto& fi = features(ki);
  const auto& fj = features(kj);
  const auto pi = extract_centers(fi).cwise_transform(to_double);
  const auto pj = extract_centers(fj).cwise_transform(to_double);

  const auto Pi = homogeneous(pi);
  const auto Pj = homogeneous(pj);

  const auto Mij_tensor = to_tensor(Mij);

  auto f_estimator = EightPointAlgorithm{};
  auto distance = EpipolarDistance{};
  auto f_err_thresh = 1e-2;

  double num_samples = 1000;
  return ransac(Mij_tensor, Pi, Pj, f_estimator, distance, num_samples,
                f_err_thresh);
}

auto check_epipolar_constraints(const Image<Rgb8>& Ii, const Image<Rgb8>& Ij,
                                const FundamentalMatrix& F,
                                const vector<Match>& Mij,
                                const Tensor_<int, 1>& sample_best)
{
  const auto scale = 0.25f;
  const auto w = int((Ii.width() + Ij.width()) * scale + 0.5f);
  const auto h = int(max(Ii.height(), Ij.height()) * scale + 0.5f);
  const auto off = sara::Point2f{float(Ii.width()), 0.f};

  if (!sara::active_window())
  {
    sara::create_window(w, h);
    sara::set_antialiasing();
  }

  if (sara::get_sizes(sara::active_window()) != Eigen::Vector2i(w, h))
    sara::resize_window(w, h);

  PairWiseDrawer drawer(Ii, Ij);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

  drawer.display_images();

  auto f_err_thresh = 1e-2;
  auto distance = EpipolarDistance{F.matrix()};

  for (size_t m = 0; m < Mij.size(); ++m)
  {
    const Vector3d X1 = Mij[m].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = Mij[m].y_pos().cast<double>().homogeneous();

    if (distance(X1, X2) > f_err_thresh)
      continue;

    if (m % 100 == 0)
    {

    drawer.draw_match(Mij[m], Blue8, false);
    get_key();

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
    }
  }

  for (int m = 0; m < sample_best.size(); ++m)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(Mij[sample_best(m)], Red8, true);

    const Vector3d X1 = Mij[sample_best(m)].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = Mij[sample_best(m)].y_pos().cast<double>().homogeneous();

    const auto proj_X1 = F.right_epipolar_line(X1);
    const auto proj_X2 = F.left_epipolar_line(X2);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Magenta8, 1);
  }

  get_key();
}

struct IndexMatch
{
  int i;
  int j;
  float score;
};

struct EpipolarEdge
{
  int i;  // left
  int j;  // right
  Matrix3d m;
};

template <>
struct CalculateH5Type<IndexMatch>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(IndexMatch)};
    INSERT_MEMBER(h5_comp_type, IndexMatch, i);
    INSERT_MEMBER(h5_comp_type, IndexMatch, j);
    INSERT_MEMBER(h5_comp_type, IndexMatch, score);
    return h5_comp_type;
  }
};

template <>
struct CalculateH5Type<EpipolarEdge>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(EpipolarEdge)};
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, i);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, j);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, m);
    return h5_comp_type;
  }
};


KeypointList<OERegion, float> read_keypoints(H5File& h5_file,
                                             const std::string& group_name)
{
  auto features = std::vector<sara::OERegion>{};
  auto descriptors = sara::Tensor_<float, 2>{};

  SARA_DEBUG << "Read features..." << std::endl;
  h5_file.read_dataset(group_name + "/" + "features", features);

  SARA_DEBUG << "Read descriptors..." << std::endl;
  h5_file.read_dataset(group_name + "/" + "descriptors", descriptors);

  return {features, descriptors};
}

auto read_matches(H5File& file, const std::string& name)
{
  auto matches = std::vector<IndexMatch>{};
  file.read_dataset(name, matches);
  return matches;
}

}  // namespace DO::Sara


GRAPHICS_MAIN()
{
#if defined(__APPLE__)
  const auto dirpath = fs::path{"/Users/david/Desktop/Datasets/sfm/castle_int"};
  auto h5_file = sara::H5File{"/Users/david/Desktop/Datasets/sfm/castle_int.h5",
                              H5F_ACC_RDWR};
#else
  const auto dirpath = fs::path{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto h5_file = sara::H5File{"/home/david/Desktop/Datasets/sfm/castle_int.h5",
                              H5F_ACC_RDWR};
#endif

  auto image_paths = sara::ls(dirpath.string(), ".png");
  std::sort(image_paths.begin(), image_paths.end());

  const auto N = int(image_paths.size());
  for (int i = 0; i < N; ++i)
  {
    for (int j = i + 1; j < N; ++j)
    {
      // Specify the image pair to process.
      const auto& fi = image_paths[i];
      const auto& fj = image_paths[j];

      const auto gi = sara::basename(fi);
      const auto gj = sara::basename(fj);

      SARA_DEBUG << gi << std::endl;
      SARA_DEBUG << gj << std::endl;
      const auto Ii = sara::imread<sara::Rgb8>(fi);
      const auto Ij = sara::imread<sara::Rgb8>(fj);


      //// Load the keypoints.
      //const auto Ki = sara::read_keypoints(h5_file, gi);
      //const auto Kj = sara::read_keypoints(h5_file, gj);
      const auto Ki = sara::compute_sift_keypoints(Ii.convert<float>());
      const auto Kj = sara::compute_sift_keypoints(Ij.convert<float>());


      // Match keypoints.
      const auto Mij = match(Ki, Kj);


      //// Save the keypoints to HDF5
      //auto Mij2 = std::vector<sara::IndexMatch>{};
      //std::transform(
      //    Mij.begin(), Mij.end(), std::back_inserter(Mij2), [](const auto& m) {
      //      return sara::IndexMatch{m.x_index(), m.y_index(), m.score()};
      //    });

      //const auto group_name = std::string{"matches"};
      //h5_file.group(group_name);

      //const auto match_dataset =
      //    group_name + "/" + std::to_string(i) + "_" + std::to_string(j);
      //h5_file.write_dataset(match_dataset, tensor_view(Mij2));


      // Estimate the fundamental matrix.
      const auto [F, num_inliers, sample_best] =
          sara::estimate_fundamental_matrix(Mij, Ki, Kj);


      // Visualize the estimated fundamental matrix.
      sara::check_epipolar_constraints(Ii, Ij, F, Mij, sample_best);
    }
  }

  return 0;
}
