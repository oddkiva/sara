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

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Logging/Logger.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/Helpers/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/Helpers/KeypointMatching.hpp>
#include <DO/Sara/SfM/Helpers/Triangulation.hpp>

#include <filesystem>


namespace fs = std::filesystem;

using namespace std::string_literals;
using namespace DO::Sara;


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

auto sara_graphics_main(int argc, char** argv) -> int
{
  auto& logger = Logger::get();

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
  const auto sift_nn_ratio = argc < 7 ? 0.6f : std::stof(argv[6]);
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
  const auto M = to_tensor(matches);
  const auto X = PointCorrespondenceList{M, u[0], u[1]};

  auto data_normalizer =
      std::make_optional(Normalizer<EssentialMatrix>{K[0], K[1]});

  SARA_LOGI(logger, "Estimating the essential matrix...");
  auto E = EssentialMatrix{};
  auto num_samples = argc < 5 ? 200 : std::stoi(argv[4]);
  auto err_thres = argc < 6 ? 0.5 : std::stod(argv[5]);
  auto inliers = Tensor_<bool, 1>{};
  auto sample_best = Tensor_<int, 1>{};
  {
    // N.B.: in my experience, the Sampson distance works less well than the
    // normal epipolar distance for the estimation of the essential matrix.
    auto inlier_predicate = InlierPredicate<SampsonEssentialEpipolarDistance>{};
    inlier_predicate.distance.K1_inv = K_inv[0];
    inlier_predicate.distance.K2_inv = K_inv[1];
    inlier_predicate.err_threshold = err_thres;

#define NISTER_METHOD
#if defined(NISTER_METHOD)
    std::cout << "WITH NISTER'S POLYNOMIAL ROOTS\n";
#else
    std::cout << "WITH STEWENIUS' GROEBNER BASIS\n";
#endif
    std::tie(E, inliers, sample_best) = ransac(  //
        X,
#if defined(NISTER_METHOD)
        NisterFivePointAlgorithm{},  //
#else
        SteweniusFivePointAlgorithm{},
#endif
        inlier_predicate, num_samples, data_normalizer, true);
  }


  // Calculate the fundamental matrix.
  print_stage("Computing the fundamental matrix...");
  auto F = FundamentalMatrix{};
  F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];

  // Extract the two-view geometry.
  SARA_LOGI(logger, "Estimating the two-view geometry...");
  auto un = u;
  std::tie(un[0], un[1]) = data_normalizer->normalize(u[0], u[1]);
  auto two_view_geometry = estimate_two_view_geometry(  //
      M, un[0], un[1], E, inliers, sample_best);

  // Filter the 3D points.
  keep_cheiral_inliers_only(two_view_geometry, inliers);

  // Add the internal camera matrices to the camera.
  two_view_geometry.C1.K = K[0];
  two_view_geometry.C2.K = K[1];
  const auto colors = extract_colors(images[0], images[1], two_view_geometry);

#if defined(__APPLE__)
  const auto geometry_h5_filepath = "/Users/oddkiva/Desktop/geometry.h5"s;
#else
  const auto geometry_h5_filepath = "/home/david/Desktop/geometry.h5"s;
#endif
  auto geometry_h5_file = H5File{geometry_h5_filepath, H5F_ACC_TRUNC};
  save_to_hdf5(geometry_h5_file, two_view_geometry, colors);
  geometry_h5_file.write_dataset("dataset_folder", data_dir.string(), true);
  geometry_h5_file.write_dataset("image_1", image_paths[0], true);
  geometry_h5_file.write_dataset("image_2", image_paths[1], true);
  geometry_h5_file.write_dataset(
      "K", (data_dir / (image_ids[1] + ".png.K")).string(), true);

  // Inspect the fundamental matrix.
  print_stage("Inspecting the fundamental matrix estimation...");
  check_epipolar_constraints(images[0], images[1], F, matches, sample_best,
                             inliers,
                             /* display_step */ 20, /* wait_key */ true);

  print_stage("Sort the points by depth...");
  const auto& geometry = two_view_geometry;
  const auto num_points = static_cast<int>(geometry.X.cols());
  const auto indices = range(num_points);

  // Retrieve the camera matrices.
  const auto P1 = geometry.C1.matrix();
  const auto P2 = geometry.C2.matrix();

  // Calculate the image coordinates from the normalized camera coordinates.
  const Eigen::MatrixXd u1 = (P1 * geometry.X).colwise().hnormalized();
  const Eigen::MatrixXd u2 = (P2 * geometry.X).colwise().hnormalized();

  using depth_t = float;
  auto points = std::vector<std::pair<int, depth_t>>{};
  for (auto i = 0; i < num_points; ++i)
    points.emplace_back(i, geometry.X.col(i).z());

  std::sort(points.begin(), points.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  display(images[0], Point2i::Zero(), 0.25);

  // The brighter the color, the further the point is.
  const auto depth_min = points.front().second;
  const auto depth_max = points.back().second;
  const auto linear = [depth_min, depth_max](auto d) {
    return (d - depth_min) / (depth_max - depth_min);
  };

  for (const auto& [index, depth] : points)
  {
    const Eigen::Vector2i ui =
        (u1.col(index) * 0.25).array().round().cast<int>();

    auto color = Rgb8{};
    color << 0, 0, static_cast<int>(linear(depth) * 255);
    if (depth < 0)
      color = Red8;  // Highlight where the problem is...
    fill_circle(ui.x(), ui.y(), 5, color);
    millisleep(1);
  }

  // The rotation is expressed in the camera coordinates.
  // But the calculation is done in the automotive/aeronautics coordinate
  // system.
  //
  // The z-coordinate of the camera coordinates is the x-axis of the automotive
  // coordinates
  //
  // clang-format off
  static const auto P = (Eigen::Matrix3d{} <<
     0,  0, 1,
    -1,  0, 0,
     0, -1, 0
  ).finished();
  // clang-format on

  const auto& R = geometry.C2.R;
  const auto& t = geometry.C2.t;
  const Eigen::Matrix3d Rw = P * R.transpose() * P.transpose();
  const Eigen::Vector3d tw = P * (-R.transpose() * t);

  // The implementation.
  const auto angles = calculate_yaw_pitch_roll(Eigen::Quaterniond{Rw});
  SARA_DEBUG << "Rw =\n" << Rw << std::endl;
  SARA_DEBUG << "tw =\n" << tw << std::endl;

  SARA_DEBUG << "yaw   = " << angles(0) * 180. / M_PI << " deg" << std::endl;
  SARA_DEBUG << "pitch = " << angles(1) * 180. / M_PI << " deg" << std::endl;
  SARA_DEBUG << "roll  = " << angles(2) * 180. / M_PI << " deg" << std::endl;

  get_key();

  return 0;
}
