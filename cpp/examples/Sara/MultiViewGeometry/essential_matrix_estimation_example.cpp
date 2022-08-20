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

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>

#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


auto calculate_yaw_pitch_roll(const Eigen::Matrix3d& R) -> Eigen::Vector3d
{
  const auto q = Eigen::Quaterniond{R};
  // roll (x-axis rotation)
  const auto sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
  const auto cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
  const auto roll = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  const auto sinp = 2 * (q.w() * q.y() - q.z() * q.x());
  const auto pitch =
      std::abs(sinp) >= 1
          ? std::copysign(M_PI / 2, sinp)  // use 90 degrees if out of range
          : std::asin(sinp);

  // yaw (z-axis rotation)
  const auto siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
  const auto cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
  const auto yaw = std::atan2(siny_cosp, cosy_cosp);

  return {yaw, pitch, roll};
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  // Use the following data structure to load images, keypoints, camera
  // parameters.
  auto views = ViewAttributes{};

  // Load images.
  print_stage("Loading images...");
  const auto data_dir =
#ifdef __APPLE__
      // "/Users/david/Desktop/Datasets/sfm/fountain_int"s;
      "/Users/david/Desktop/Datasets/sfm/castle_int"s;
#else
      // "/home/david/Desktop/Datasets/sfm/fountain_int"s;
      "/home/david/Desktop/Datasets/sfm/castle_int"s;
#endif
  const auto image_id1 = std::string{argv[1]};  // "0005"s;
  const auto image_id2 = std::string{argv[2]};  // "0004"s;
  views.image_paths = {
      data_dir + "/" + image_id1 + ".png",
      data_dir + "/" + image_id2 + ".png",
  };
  views.read_images();
  SARA_CHECK(views.images[0].sizes().transpose());
  SARA_CHECK(views.images[1].sizes().transpose());


  print_stage("Loading the internal camera matrices...");
  views.cameras.resize(2 /* views */);
  views.cameras[0].K =
      read_internal_camera_parameters(data_dir + "/" + image_id1 + ".png.K")
          .cast<double>();
  views.cameras[1].K =
      read_internal_camera_parameters(data_dir + "/" + image_id2 + ".png.K")
          .cast<double>();
  SARA_CHECK(views.cameras[0].K);
  SARA_CHECK(views.cameras[1].K);


  print_stage("Computing keypoints...");
  const auto image_pyr_params = ImagePyramidParams(-1);
  views.keypoints = {compute_sift_keypoints(views.images[0].convert<float>(),
                                            image_pyr_params),
                     compute_sift_keypoints(views.images[1].convert<float>(),
                                            image_pyr_params)};

  // Use the following data structures to store the epipolar geometry data.
  auto epipolar_edges = EpipolarEdgeAttributes{};
  epipolar_edges.initialize_edges(2 /* views */);
  epipolar_edges.resize_fundamental_edge_list();
  epipolar_edges.resize_essential_edge_list();


  print_stage("Matching keypoints...");
  const auto sift_nn_ratio = argc < 6 ? 0.6f : std::stof(argv[5]);
  epipolar_edges.matches = {
      match(views.keypoints[0], views.keypoints[1], sift_nn_ratio)};
  const auto& matches = epipolar_edges.matches[0];


  print_stage("Performing data transformations...");
  // Invert the internal camera matrices.
  const auto K_inv = std::array<Matrix3d, 2>{views.cameras[0].K.inverse(),
                                             views.cameras[1].K.inverse()};
  // Tensors of image coordinates.
  const auto& f0 = features(views.keypoints[0]);
  const auto& f1 = features(views.keypoints[1]);
  const auto u = std::array{homogeneous(extract_centers(f0)).cast<double>(),
                            homogeneous(extract_centers(f1)).cast<double>()};
  // Tensors of camera coordinates.
  const auto un = std::array{apply_transform(K_inv[0], u[0]),
                             apply_transform(K_inv[1], u[1])};
  static_assert(std::is_same_v<decltype(un[0]), const Tensor_<double, 2>&>);
  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  const auto M = to_tensor(matches);


  print_stage("Estimating the essential matrix...");
  auto& E = epipolar_edges.E[0];
  auto& num_samples = epipolar_edges.E_num_samples[0];
  auto& err_thres = epipolar_edges.E_noise[0];
  auto& inliers = epipolar_edges.E_inliers[0];
  auto sample_best = Tensor_<int, 1>{};
  auto estimator = NisterFivePointAlgorithm{};
  auto distance = EpipolarDistance{};
  {
    num_samples = argc < 4 ? 200 : std::stoi(argv[3]);
    err_thres = argc < 5 ? 1e-3 : std::stod(argv[4]);
    std::tie(E, inliers, sample_best) =
        ransac(M, un[0], un[1], estimator, distance, num_samples, err_thres);
    E.matrix() = E.matrix().normalized();

    epipolar_edges.E_inliers[0] = inliers;
    epipolar_edges.E_best_samples[0] = sample_best;
  }


  // Calculate the fundamental matrix.
  print_stage("Computing the fundamental matrix...");
  auto& F = epipolar_edges.F[0];
  {
    F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];

    epipolar_edges.F_num_samples[0] = 1000;
    epipolar_edges.F_noise = epipolar_edges.E_noise;
    epipolar_edges.F_inliers = epipolar_edges.E_inliers;
    epipolar_edges.F_best_samples = epipolar_edges.E_best_samples;
  }

  // Extract the two-view geometry.
  print_stage("Estimating the two-view geometry...");
  epipolar_edges.two_view_geometries = {
      estimate_two_view_geometry(M, un[0], un[1], E, inliers, sample_best)};

  // Filter the 3D points.
  auto& two_view_geometry = epipolar_edges.two_view_geometries.front();
  keep_cheiral_inliers_only(two_view_geometry, inliers);

  // Add the internal camera matrices to the camera.
  two_view_geometry.C1.K = views.cameras[0].K;
  two_view_geometry.C2.K = views.cameras[1].K;
  auto colors = extract_colors(views.images[0],  //
                               views.images[1],  //
                               two_view_geometry);
  save_to_hdf5(two_view_geometry, colors);

  // Inspect the fundamental matrix.
  print_stage("Inspecting the fundamental matrix estimation...");
  check_epipolar_constraints(views.images[0], views.images[1], F, matches,
                             sample_best, inliers,
                             /* display_step */ 20, /* wait_key */ true);

  print_stage("Sort the points by depth...");
  const auto& geometry = two_view_geometry;
  const auto num_points = static_cast<int>(geometry.X.cols());
  const auto indices = range(num_points);

  // Retrieve the camera matrices.
  const auto P1 = geometry.C1.matrix();
  const auto P2 = geometry.C2.matrix();

  // Calculate the image coordinates from the normalized camera coordinates.
  const MatrixXd u1 = (P1 * geometry.X).colwise().hnormalized();
  const MatrixXd u2 = (P2 * geometry.X).colwise().hnormalized();

  using depth_t = float;
  auto points = std::vector<std::pair<int, depth_t>>{};
  for (auto i = 0; i < num_points; ++i)
    points.emplace_back(i, geometry.X.col(i).z());

  std::sort(points.begin(), points.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  display(views.images[0], Point2i::Zero(), 0.25);

  // The brighter the color, the further the point is.
  const auto depth_min = points.front().second;
  const auto depth_max = points.back().second;
  const auto linear = [depth_min, depth_max](auto d) {
    return (d - depth_min) / (depth_max - depth_min);
  };

  for (const auto& [index, depth] : points)
  {
    const Eigen::Vector2d ui = u1.col(index) * 0.25;

    auto color = Rgb8{};
    color << 0, 0, int(linear(depth) * 255);
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
  const auto angles = calculate_yaw_pitch_roll(Rw);
  SARA_DEBUG << "Rw =\n" << Rw << std::endl;
  SARA_DEBUG << "tw =\n" << tw << std::endl;

  SARA_DEBUG << "yaw   = " << angles(0) * 180. / M_PI << " deg" << std::endl;
  SARA_DEBUG << "pitch = " << angles(1) * 180. / M_PI << " deg" << std::endl;
  SARA_DEBUG << "roll  = " << angles(2) * 180. / M_PI << " deg" << std::endl;

  get_key();

  return 0;
}
