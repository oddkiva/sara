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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>

#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


// FIXME: not valid in the camera reference coordinate.
auto calculate_yaw_pitch_roll(const Eigen::Matrix3d& R) -> Eigen::Vector3d
{
  auto angles = Eigen::Vector3d{};
  // Pitch angle.
  angles(1) = -std::asin(R(2, 0));

  // Gymbal lock: pitch = -90
  if (R(2, 0) == 1)
  {
    angles(0) = 0;                               // yaw = 0
    angles(2) = std::atan2(-R(0, 1), -R(0, 2));  // roll
    std::cout << "Gimbal lock: pitch = -90" << std::endl;
  }

  // Gymbal lock: pitch = 90
  else if (R(2, 0) == -1)
  {
    angles(0) = 0;                             // yaw = 0
    angles(2) = std::atan2(R(0, 1), R(0, 2));  // roll
    std::cout << "Gimbal lock: pitch = +90" << std::endl;
  }

  // General solution
  else
  {
    angles(0) = std::atan2(R(1, 0), R(0, 0));
    angles(2) = std::atan2(R(2, 1), R(2, 2));
  }

  return angles;  // Euler angles in order yaw, pitch, roll
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int, char** argv)
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
      "/home/david/Desktop/Datasets/sfm/castle_int"s;
      // "/home/david/Desktop/Datasets/sfm/fountain_int"s;
#endif
  const auto image_id1 = std::string{argv[1]};  // "0005"s;
  const auto image_id2 = std::string{argv[2]};  // "0004"s;
  views.image_paths = {
      data_dir + "/" + image_id1 + ".png",
      data_dir + "/" + image_id2 + ".png",
  };
  views.read_images();


  print_stage("Loading the internal camera matrices...");
  views.cameras.resize(2 /* views */);
  views.cameras[0].K =
      read_internal_camera_parameters(data_dir + "/" + image_id1 + ".png.K")
          .cast<double>();
  views.cameras[1].K =
      read_internal_camera_parameters(data_dir + "/" + image_id2 + ".png.K")
          .cast<double>();


  print_stage("Computing keypoints...");
  const auto image_pyr_params = ImagePyramidParams(0);
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
  epipolar_edges.matches = {match(views.keypoints[0], views.keypoints[1])};
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
    num_samples = 200;
    err_thres = 1e-3;
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

  display(views.images[0], 0, 0, 0.25);

  // The brighter the color, the further the point is.
  const auto depth_min = points.front().second;
  const auto depth_max = points.back().second;
  const auto linear = [depth_min, depth_max](auto d) {
    return (d - depth_min) / (depth_max - depth_min);
  };

  for (const auto& [index, depth] : points)
  {
    SARA_DEBUG << depth << std::endl;
    const Eigen::Vector2d ui = u1.col(index) * 0.25;

    auto color = Rgb8{};
    color << 0, 0, int(linear(depth) * 255);
    fill_circle(ui.x(), ui.y(), 5, color);
    millisleep(1);
  }

  const auto& R = geometry.C2.R;
  const auto& t = geometry.C2.t;
  const Eigen::Matrix3d Rw = R.transpose();
  const Eigen::Vector3d tw = -R.transpose() * t;
  SARA_DEBUG << "Rw =\n" << Rw << std::endl;
  SARA_DEBUG << "tw =\n" << tw << std::endl;

  SARA_DEBUG << "yaw pitch roll =\n"
             << Rw.eulerAngles(1, 0, 2) * 180. / M_PI << std::endl;
             // << calculate_yaw_pitch_roll(Rw) * 180. / M_PI << std::endl;

  get_key();

  return 0;
}
