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

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>

#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>


using namespace std;
using namespace std::string_literals;
using namespace DO::Sara;


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

int sara_graphics_main(int argc, char** argv)
{
  // Use the following data structure to load images, keypoints, camera
  // parameters.
  auto views = ViewAttributes{};

  // Load images.
  print_stage("Loading images...");
  const auto data_dir = argc < 2
                            ? "/Users/david/Desktop/Datasets/sfm/castle_int"s
                            : std::string{argv[1]};
  const auto image_id1 = std::string{argv[2]};
  const auto image_id2 = std::string{argv[3]};
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
  const auto sift_nn_ratio = argc < 5 ? 0.6f : std::stof(argv[4]);
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
  auto un = std::array{apply_transform(K_inv[0], u[0]),
                       apply_transform(K_inv[1], u[1])};
  // Normalize backprojected rays to unit norm.
  for (auto i = 0u; i < un.size(); ++i)
    un[i].colmajor_view().matrix().colwise().normalize();


  // List the matches as a 2D-tensor where each row encodes a match 'm' as a
  // pair of point indices (i, j).
  print_stage("Estimating the two view geometry...");
  const auto M = to_tensor(matches);
  const auto X = PointCorrespondenceList{M, un[0], un[1]};

  const auto num_samples = argc < 6 ? 1000 : std::stoi(argv[5]);
  const auto err_thres = argc < 7 ? 5e-3 : std::stod(argv[6]);

  auto solver = RelativePoseSolver<NisterFivePointAlgorithm>{
      CheiralityCriterion::CHEIRAL_COMPLETE};

  auto inlier_predicate = CheiralAndEpipolarConsistency{};
  inlier_predicate.err_threshold = err_thres;

  auto [geometry, inliers, sample_best] =
      ransac(X, solver, inlier_predicate, num_samples, std::nullopt, true);

  SARA_DEBUG << "Geometry =\n" << geometry << std::endl;
  SARA_DEBUG << "inliers count = " << inliers.flat_array().count() << std::endl;
  SARA_DEBUG << "Num 3D points = " << geometry.X.cols() << std::endl;

  // Retrieve the essential matrix.
  print_stage("Saving the data from the essential matrix estimation...");
  epipolar_edges.E[0] = essential_matrix(geometry.C2.R, geometry.C2.t);
  epipolar_edges.E_num_samples[0] = num_samples;
  epipolar_edges.E_noise[0] = err_thres;
  epipolar_edges.E_inliers[0] = inliers;


  // Retrieve the fundamental matrix.
  print_stage("Saving the fundamental matrix...");
  auto& F = epipolar_edges.F[0];
  {
    const auto& E = epipolar_edges.E[0];
    F.matrix() = K_inv[1].transpose() * E.matrix() * K_inv[0];

    epipolar_edges.F_num_samples[0] = num_samples;
    epipolar_edges.F_noise = epipolar_edges.E_noise;
    epipolar_edges.F_inliers = epipolar_edges.E_inliers;
  }

  // Retrieve all the 3D points by triangulation.
  print_stage("Retriangulating the inliers...");
  auto& X3d = geometry.X;
  auto& s1 = geometry.scales1;
  auto& s2 = geometry.scales2;
  X3d.resize(4, inliers.flat_array().count());
  s1.resize(inliers.flat_array().count());
  s2.resize(inliers.flat_array().count());
  for (auto i = 0, j = 0; i < inliers.size(0); ++i)
  {
    if (!inliers(i))
      continue;

    const Eigen::Vector3d u1 = X[i][0].vector();
    const Eigen::Vector3d u2 = X[i][1].vector();
    const auto [Xj, s1j, s2j] = triangulate_single_point_linear_eigen(
        geometry.C1.matrix(), geometry.C2.matrix(), u1, u2);

    X3d.col(j) = Xj;
    s1(j) = s1j;
    s2(j) = s2j;
    if (X3d.col(j)(2) <= 0)
      std::cout << j << " -> " << X3d.col(j).transpose() << std::endl;
    ++j;

  }

  if (!(X3d.row(2).array() > 0).all())
    throw std::runtime_error{"Uh Oh.... Cheirality is wrong!"};


  return 0;
}
