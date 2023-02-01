// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors/SIFT.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/BrownConradyCamera.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/OmnidirectionalCamera.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/InlierPredicates.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/RelativePoseSolver.hpp>
#include <DO/Sara/MultiViewGeometry/Miscellaneous.hpp>
#include <DO/Sara/RANSAC/RANSACv2.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>

#include "ImageWarpUtilities.hpp"

#include <filesystem>

#include <omp.h>


namespace fs = std::filesystem;
namespace sara = DO::Sara;


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

auto sara_graphics_main(int, char**) -> int
{
  const auto num_threads = omp_get_max_threads();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads);

  using namespace std::string_literals;
#if defined(__APPLE__)
  const auto video_path = "/Users/david/Desktop/Datasets/sample-1.mp4"s;
#else
  const auto video_path = "/home/david/Desktop/Datasets/sample-1.mp4"s;
#endif

  auto camera = sara::v2::BrownConradyDistortionModel<double>{};
  camera.fx() = 917.2878392016245;
  camera.fy() = 917.2878392016245;
  camera.shear() = 0.;
  camera.u0() = 960.;
  camera.v0() = 540.;
  camera.k() << -0.2338367557617234, 0.05952465745165465, -0.007947847982157091;
  camera.p() << -0.0003137658969742134, 0.00021943576376532096;

  const auto K = camera.calibration_matrix();
  const Eigen::Matrix3d K_inv = K.inverse();
  SARA_DEBUG << "K =\n" << K << std::endl;
  SARA_DEBUG << "K^{-1} =\n" << K_inv << std::endl;

  auto video_stream = sara::VideoStream{video_path};

  auto frame_rgb8 = video_stream.frame();
  auto frame_gray32f = sara::Image<float>{video_stream.sizes()};

  auto frames_rgb_undist = std::array<sara::Image<sara::Rgb8>, 2>{};
  auto frames_work = std::array<sara::Image<float>, 2>{};
  auto& frame_undistorted = frames_work[1];

  const auto coords_map = undistortion_map(camera, video_stream.sizes());
  const auto& [umap, vmap] = coords_map;

  constexpr auto sift_nn_ratio = 0.6f;
  constexpr auto ransac_iterations_max = 200;
  constexpr auto ransac_confidence = 0.999;
  constexpr auto err_thres = 2.;
  const auto image_pyr_params = sara::ImagePyramidParams(0);
  auto solver = sara::RelativePoseSolver<sara::NisterFivePointAlgorithm>{};
  auto inlier_predicate = sara::CheiralAndEpipolarConsistency{};
  {
    inlier_predicate.distance.K1_inv = K_inv;
    inlier_predicate.distance.K2_inv = K_inv;
    inlier_predicate.err_threshold = err_thres;
  }

  auto keys = std::array<sara::KeypointList<sara::OERegion, float>, 2>{};
  auto E = sara::EssentialMatrix{};
  auto F = sara::FundamentalMatrix{};
  auto matches = std::vector<sara::Match>{};
  auto inliers = sara::Tensor_<bool, 1>{};
  auto sample_best = sara::Tensor_<int, 1>{};
  auto geometry = sara::TwoViewGeometry{};


  sara::create_window(video_stream.sizes());
  sara::set_antialiasing();

  auto frame_index = -1;
  while (video_stream.read())
  {
    ++frame_index;
    if (frame_index % 5 != 0)
      continue;
    SARA_CHECK(frame_index);

    frames_rgb_undist[0].swap(frames_rgb_undist[1]);
    frames_work[0].swap(frames_work[1]);

    if (frame_undistorted.sizes() != frame_rgb8.sizes())
      frame_undistorted.resize(frame_rgb8.sizes());
    if (frames_rgb_undist[1].sizes() != frame_rgb8.sizes())
      frames_rgb_undist[1].resize(frame_rgb8.sizes());
    sara::from_rgb8_to_gray32f(frame_rgb8, frame_gray32f);
    warp(umap, vmap, frame_rgb8, frames_rgb_undist[1]);
    warp(umap, vmap, frame_gray32f, frame_undistorted);
    auto display = frames_rgb_undist[1];

    sara::print_stage("Computing keypoints...");
    std::swap(keys[0], keys[1]);
    keys[1] = sara::compute_sift_keypoints(frame_undistorted, image_pyr_params);

    const auto& f0 = sara::features(keys[0]);
    const auto& f1 = sara::features(keys[1]);
    const auto do_relative_pose_estimation = !f0.empty() && !f1.empty();

    if (do_relative_pose_estimation)
    {
      sara::print_stage("Matching keypoints...");
      matches = match(keys[0], keys[1], sift_nn_ratio);
      // Put a hard limit of 1000 matches to scale.
      if (matches.size() > 1000)
        matches.resize(1000);

      sara::print_stage("Estimating the relative pose...");
      const auto u = std::array{
          sara::homogeneous(sara::extract_centers(f0)).cast<double>(),
          sara::homogeneous(sara::extract_centers(f1)).cast<double>()};
      // List the matches as a 2D-tensor where each row encodes a match 'm' as a
      // pair of point indices (i, j).
      const auto M = sara::to_tensor(matches);
      const auto X = sara::PointCorrespondenceList{M, u[0], u[1]};
      auto data_normalizer =
          std::make_optional(sara::Normalizer<sara::TwoViewGeometry>{K, K});
      auto sample_best = sara::Tensor_<int, 1>{};
      std::tie(geometry, inliers, sample_best) =
          sara::v2::ransac(X, solver, inlier_predicate, ransac_iterations_max,
                           ransac_confidence, data_normalizer, true);
      SARA_DEBUG << "Geometry =\n" << geometry << std::endl;
      SARA_DEBUG << "inliers count = " << inliers.flat_array().count()
                 << std::endl;

      // Retrieve all the 3D points by triangulation.
      sara::print_stage("Retriangulating the inliers...");
      auto& points = geometry.X;
      auto& s1 = geometry.scales1;
      auto& s2 = geometry.scales2;
      points.resize(4, inliers.flat_array().count());
      s1.resize(inliers.flat_array().count());
      s2.resize(inliers.flat_array().count());
      auto cheiral_inlier_count = 0;
      {
        auto& j = cheiral_inlier_count;
        for (auto i = 0; i < inliers.size(0); ++i)
        {
          if (!inliers(i))
            continue;

          const Eigen::Vector3d u1 = K_inv * X[i][0].vector();
          const Eigen::Vector3d u2 = K_inv * X[i][1].vector();
          const auto [Xj, s1j, s2j] =
              sara::triangulate_single_point_linear_eigen(
                  geometry.C1.matrix(), geometry.C2.matrix(), u1, u2);
          const auto cheiral = s1j > 0 && s2j > 0;
          if (!cheiral)
            continue;

          // Also we want z in [0, 200] meters max...
          // We want to avoid 3D point corresponding to the sky...
          Eigen::Vector4d Xjh = Xj.hnormalized().homogeneous();
          const auto reasonable = 0 < Xjh.z() && Xjh.z() < 200;
          if (!reasonable)
            continue;

          points.col(j) = Xjh;
          s1(j) = s1j;
          s2(j) = s2j;
          ++j;
        }
        SARA_CHECK(cheiral_inlier_count);
        points = points.leftCols(cheiral_inlier_count);
        s1 = s1.head(cheiral_inlier_count);
        s2 = s2.head(cheiral_inlier_count);
      }

      sara::print_stage("Extracting the colors...");
      geometry.C1.K = K;
      geometry.C2.K = K;
      auto colors = sara::extract_colors(frames_rgb_undist[0],
                                         frames_rgb_undist[1],  //
                                         geometry);

      sara::print_stage("Saving to HDF5");
      {
#if defined(__APPLE__)
        const auto geometry_h5_filepath = "/Users/david/Desktop/geometry" +
#else
        const auto geometry_h5_filepath = "/home/david/Desktop/geometry" +
#endif
                                          std::to_string(frame_index) + ".h5"s;
        auto geometry_h5_file =
            sara::H5File{geometry_h5_filepath, H5F_ACC_TRUNC};
        sara::save_to_hdf5(geometry_h5_file, geometry, colors);
      }

      F.matrix() = K_inv.transpose() * E.matrix() * K_inv;

      sara::print_stage("Draw...");
      for (auto m = 0u; m < matches.size(); ++m)
      {
        if (!inliers(m))
          continue;
        const auto& match = matches[m];
        sara::draw(display, match.x(), sara::Blue8);
        sara::draw(display, match.y(), sara::Cyan8);
        sara::draw_arrow(display, match.x_pos(), match.y_pos(), sara::Yellow8);
      }

      sara::display(display);
      // if (sara::get_key() == sara::KEY_ESCAPE)
      //   break;
    }
  }

  return 0;
}
