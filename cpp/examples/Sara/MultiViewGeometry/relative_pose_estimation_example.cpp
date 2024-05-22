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
//! This program parses Strecha's datasets.
//!
//! Recovering the pose immediately after the estimated essential matrix and
//! then counting the cheiral triangulated points does not work well.
//! - This is slow because of the triangulation.
//!
//! Estimating the essential matrix first works and counting the inliers
//! without worrying whether the triangulated points are cheiral just works
//! better. Then after the pose recovery at the end just gives better results.

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/Helpers.hpp>

#include <filesystem>


using namespace std::string_literals;
using namespace DO::Sara;

namespace fs = std::filesystem;


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

int sara_graphics_main(int argc, char** argv)
{
  auto& logger = Logger::get();

  // Load images.
  SARA_LOGI(logger, "Loading images...");
  const auto data_dir =
      argc < 2 ? fs::path{"/Users/oddkiva/Desktop/datasets/sfm/fountain_int"}
               : fs::path{argv[1]};
  const auto image_ids = std::array<std::string, 2>{
      argc < 3 ? "0000" : argv[2],
      argc < 4 ? "0001" : argv[3],
  };
  const auto image_paths = std::array{
      (data_dir / (image_ids[0] + ".png")).string(),
      (data_dir / (image_ids[1] + ".png")).string()  //
  };
  const auto images = std::array{
      imread<Rgb8>(image_paths[0]),  //
      imread<Rgb8>(image_paths[1])   //
  };

  SARA_LOGI(logger, "Loading the internal camera matrices...");
  const auto K = std::array{
      read_internal_camera_parameters(
          (data_dir / (image_ids[0] + ".png.K")).string())
          .cast<double>(),
      read_internal_camera_parameters(
          (data_dir / (image_ids[1] + ".png.K")).string())
          .cast<double>()  //
  };
  for (auto i = 0; i < 2; ++i)
    SARA_LOGD(logger, "K[{}] =\n{}", i, K[i]);

  SARA_LOGI(logger, "Computing keypoints...");
  const auto image_pyr_params = ImagePyramidParams(-1);
  const auto keypoints = std::array{
      compute_sift_keypoints(images[0].convert<float>(), image_pyr_params),
      compute_sift_keypoints(images[1].convert<float>(), image_pyr_params)  //
  };

  // Use the following data structures to store the epipolar geometry data.
  SARA_LOGI(logger, "Matching keypoints...");
  const auto sift_nn_ratio = argc < 5 ? 0.6f : std::stof(argv[4]);
  const auto matches = match(keypoints[0], keypoints[1], sift_nn_ratio);


  SARA_LOGI(logger, "Performing data transformations...");
  // Invert the internal camera matrices.
  const auto K_inv = std::array<Eigen::Matrix3d, 2>{
      K[0].inverse(),
      K[1].inverse()  //
  };
  // Tensors of image coordinates.
  const auto& f0 = features(keypoints[0]);
  const auto& f1 = features(keypoints[1]);
  const auto u = std::array{homogeneous(extract_centers(f0)).cast<double>(),
                            homogeneous(extract_centers(f1)).cast<double>()};

  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  SARA_LOGI(logger, "Estimating the two view geometry...");
  const auto M = to_tensor(matches);
  const auto X = PointCorrespondenceList{M, u[0], u[1]};
  const auto num_samples = argc < 6 ? 1000 : std::stoi(argv[5]);
  const auto err_thres = argc < 7 ? 2. : std::stod(argv[6]);

  auto solver = RelativePoseSolver<NisterFivePointAlgorithm>{};

  auto data_normalizer =
      std::make_optional(Normalizer<TwoViewGeometry>{K[0], K[1]});

  auto inlier_predicate = CheiralAndEpipolarConsistency{};
  inlier_predicate.distance.K1_inv = K_inv[0];
  inlier_predicate.distance.K2_inv = K_inv[0];
  inlier_predicate.err_threshold = err_thres;

  auto [geometry, inliers, sample_best] =
      ransac(X, solver, inlier_predicate, num_samples, data_normalizer, true);

  SARA_DEBUG << "Geometry =\n" << geometry << std::endl;
  SARA_DEBUG << "inliers count = " << inliers.flat_array().count() << std::endl;

  // Retrieve the essential matrix.
  SARA_LOGI(logger, "Saving the data from the essential matrix estimation...");
  const auto E = essential_matrix(geometry.C2.R, geometry.C2.t);

  // Retrieve the fundamental matrix.
  SARA_LOGI(logger, "Saving the fundamental matrix...");
  auto F = FundamentalMatrix{};
  F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];

  // Retrieve all the 3D points by triangulation.
  SARA_LOGI(logger, "Retriangulating the inliers...");
  auto& points = geometry.X;
  auto& s1 = geometry.scales1;
  auto& s2 = geometry.scales2;
  points.resize(4, inliers.flat_array().count());
  s1.resize(inliers.flat_array().count());
  s2.resize(inliers.flat_array().count());
  for (auto i = 0, j = 0; i < inliers.size(0); ++i)
  {
    if (!inliers(i))
      continue;

    const Eigen::Vector3d u1 = K_inv[0] * X[i][0].vector();
    const Eigen::Vector3d u2 = K_inv[1] * X[i][1].vector();
    const auto [Xj, s1j, s2j] = triangulate_single_point_linear_eigen(
        geometry.C1.matrix(), geometry.C2.matrix(), u1, u2);
    const auto cheiral = s1j > 0 && s2j > 0;
    if (!cheiral)
      continue;

    points.col(j) = Xj;
    s1(j) = s1j;
    s2(j) = s2j;
    ++j;
  }
  if (!(s1.array() > 0 && s2.array() > 0).all())
    throw std::runtime_error{"Uh Oh.... Cheirality is wrong!"};

  // Add the internal camera matrices to the camera.
  geometry.C1.K = K[0];
  geometry.C2.K = K[1];
  auto colors = extract_colors(images[0], images[1], geometry);

#if defined(__APPLE__)
  const auto geometry_h5_filepath = "/Users/oddkiva/Desktop/geometry.h5"s;
#else
  const auto geometry_h5_filepath = "/home/david/Desktop/geometry.h5"s;
#endif
  auto geometry_h5_file = H5File{geometry_h5_filepath, H5F_ACC_TRUNC};
  save_to_hdf5(geometry_h5_file, geometry, colors);
  geometry_h5_file.write_dataset("dataset_folder", data_dir, true);
  geometry_h5_file.write_dataset("image_1", image_paths[0], true);
  geometry_h5_file.write_dataset("image_2", image_paths[1], true);
  geometry_h5_file.write_dataset(
      "K", (data_dir / (image_ids[1] + ".png.K")).string(), true);

  // Inspect the fundamental matrix.
  SARA_LOGI(logger, "Inspecting the fundamental matrix estimation...");
  check_epipolar_constraints(images[0], images[1], F, matches, sample_best,
                             inliers,
                             /* display_step */ 20, /* wait_key */ true);

  SARA_LOGI(logger, "Sort the points by depth...");
  const auto num_points = static_cast<int>(geometry.X.cols());
  const auto indices = range(num_points);

  // Retrieve the camera matrices.
  const auto P1 = geometry.C1.matrix();
  const auto P2 = geometry.C2.matrix();

  // Calculate the image coordinates from the normalized camera coordinates.
  const Eigen::MatrixXd u1 = (P1 * geometry.X).colwise().hnormalized();
  const Eigen::MatrixXd u2 = (P2 * geometry.X).colwise().hnormalized();

  using depth_t = float;
  auto depths = std::vector<std::pair<int, depth_t>>{};
  for (auto i = 0; i < num_points; ++i)
    depths.emplace_back(i, geometry.X.col(i).z());

  std::sort(depths.begin(), depths.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  display(images[0], Point2i::Zero(), 0.25);

  // The brighter the color, the further the point is.
  const auto depth_min = depths.front().second;
  const auto depth_max = depths.back().second;
  const auto linear = [depth_min, depth_max](auto d) {
    return (d - depth_min) / (depth_max - depth_min);
  };

  for (const auto& [index, depth] : depths)
  {
    const Eigen::Vector2i ui =
        (u1.col(index) * 0.25).array().round().matrix().cast<int>();

    auto color = Rgb8{};
    color << 0, 0, int(linear(depth) * 255);
    if (depth < 0)
      color = Red8;  // Highlight where the problem is...
    fill_circle(ui.x(), ui.y(), 5, color);
    millisleep(1);
  }

  get_key();

  return 0;
}
