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
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointMatching.hpp>
#include <DO/Sara/SfM/BuildingBlocks/Triangulation.hpp>
#include <DO/Sara/VideoIO.hpp>
#include <DO/Sara/Visualization/Features/Draw.hpp>

#define BROWN_CONRADY


namespace sara = DO::Sara;

template <typename CameraModel>
auto undistortion_map(const CameraModel& camera, const Eigen::Vector2i& sizes)
    -> std::array<sara::Image<float>, 2>
{
  auto coords_map = std::array<sara::Image<float>, 2>{};
  for (auto& coord_map : coords_map)
    coord_map.resize(sizes);
  auto& [umap, vmap] = coords_map;

  const auto& w = sizes.x();
  const auto& h = sizes.y();

  for (int v = 0; v < h; ++v)
  {
    for (int u = 0; u < w; ++u)
    {
      // Backproject the pixel from the destination camera plane.
      const auto uv = Eigen::Vector2d(u, v);
      const Eigen::Vector2f uvd = camera.distort(uv).template cast<float>();

      umap(u, v) = uvd.x();
      vmap(u, v) = uvd.y();
    }
  }

  return coords_map;
}

auto warp(const sara::ImageView<float>& u_map,  //
          const sara::ImageView<float>& v_map,
          const sara::ImageView<float>& frame,
          sara::ImageView<float>& frame_warped)
{
  const auto w = frame.width();
  const auto h = frame.height();
  const auto wh = w * h;

#pragma omp parallel for
  for (auto p = 0; p < wh; ++p)
  {
    // Destination pixel.
    const auto y = p / w;
    const auto x = p - w * y;

    auto xyd = Eigen::Vector2d{};
    xyd << u_map(x, y), v_map(x, y);

    const auto in_image_domain = 0 <= xyd.x() && xyd.x() < w - 1 &&  //
                                 0 <= xyd.y() && xyd.y() < h - 1;
    if (!in_image_domain)
    {
      frame_warped(x, y) = 0.f;
      continue;
    }

    const auto color = sara::interpolate(frame, xyd);
    frame_warped(x, y) = color;
  }
}

auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}

auto sara_graphics_main(int, char**) -> int
{
  using namespace std::string_literals;
  const auto video_path = "/Users/david/Desktop/Datasets/sample-1.mp4"s;

#if defined(BROWN_CONRADY)
  auto camera = sara::v2::BrownConradyDistortionModel<double>{};
#else
  auto camera = sara::v2::OmnidirectionalCamera<double>{};
#endif
  camera.fx() = 917.2878392016245;
  camera.fy() = 917.2878392016245;
  camera.shear() = 0.;
  camera.u0() = 960.;
  camera.v0() = 540.;
#if defined(BROWN_CONRADY)
  camera.k() << -0.2338367557617234, 0.05952465745165465, -0.007947847982157091;
#else
  camera.k() << -0.2338367557617234, 0.05952465745165465;
#endif
  camera.p() << -0.0003137658969742134, 0.00021943576376532096;
#if !defined(BROWN_CONRADY)
  camera.xi() = 0.;
#endif

  const auto K = camera.calibration_matrix();
  const Eigen::Matrix3d K_inv = K.inverse();

  auto video_stream = sara::VideoStream{video_path};

  auto frame_rgb8 = video_stream.frame();
  auto frame_gray32f = sara::Image<float>{video_stream.sizes()};

  auto frames_work = std::array<sara::Image<float>, 2>{};
  auto& frame_undistorted = frames_work[1];

  const auto coords_map = undistortion_map(camera, video_stream.sizes());
  const auto& [umap, vmap] = coords_map;

  // Use the following data structure to load images, keypoints, camera
  // parameters.
  auto views = sara::ViewAttributes{};
  views.keypoints.resize(2);

  auto epipolar_edges = sara::EpipolarEdgeAttributes{};
  epipolar_edges.initialize_edges(2 /* views */);
  epipolar_edges.resize_fundamental_edge_list();
  epipolar_edges.resize_essential_edge_list();

  sara::create_window(video_stream.sizes());
  sara::set_antialiasing();
  while (video_stream.read())
  {
    frames_work[0].swap(frames_work[1]);

    if (frame_undistorted.sizes() != frame_rgb8.sizes())
      frame_undistorted.resize(frame_rgb8.sizes());
    sara::from_rgb8_to_gray32f(frame_rgb8, frame_gray32f);
    warp(umap, vmap, frame_gray32f, frame_undistorted);
    sara::display(frame_undistorted);

    sara::print_stage("Computing keypoints...");
    const auto image_pyr_params = sara::ImagePyramidParams(0);
    std::swap(sara::features(views.keypoints[0]),
              sara::features(views.keypoints[1]));
    std::swap(sara::descriptors(views.keypoints[0]),
              sara::descriptors(views.keypoints[1]));
    views.keypoints[1] =
        sara::compute_sift_keypoints(frame_undistorted, image_pyr_params);

    const auto& f0 = sara::features(views.keypoints[0]);
    const auto& f1 = sara::features(views.keypoints[1]);
    const auto do_relative_pose_estimation = !f0.empty() && !f1.empty();
    auto& matches = epipolar_edges.matches[0];
    auto& inliers = epipolar_edges.E_inliers[0];

    if (do_relative_pose_estimation)
    {
      sara::print_stage("Matching keypoints...");
      static constexpr auto sift_nn_ratio = 0.6f;
      matches = match(views.keypoints[0], views.keypoints[1], sift_nn_ratio);

      sara::print_stage("Converting the keypoints to homogeneous coordinates...");
      const auto u = std::array{
          sara::homogeneous(sara::extract_centers(f0)).cast<double>(),
          sara::homogeneous(sara::extract_centers(f1)).cast<double>()};
      sara::print_stage("Preparing matches for epipolar geometry...");
      // List the matches as a 2D-tensor where each row encodes a match 'm' as a
      // pair of point indices (i, j).
      const auto M = sara::to_tensor(matches);

      const auto X = sara::PointCorrespondenceList{M, u[0], u[1]};
      auto data_normalizer =
          std::make_optional(sara::Normalizer<sara::EssentialMatrix>{K, K});

      sara::print_stage("Estimating the essential matrix...");
      auto& E = epipolar_edges.E[0];
      auto& num_samples = epipolar_edges.E_num_samples[0];
      auto& err_thres = epipolar_edges.E_noise[0];
      auto sample_best = sara::Tensor_<int, 1>{};
      {
        num_samples = 200;
        err_thres = 2.;

        auto inlier_predicate =
            sara::InlierPredicate<sara::SampsonEssentialEpipolarDistance>{};
        inlier_predicate.distance.K1_inv = K_inv;
        inlier_predicate.distance.K2_inv = K_inv;
        inlier_predicate.err_threshold = err_thres;

        std::tie(E, inliers, sample_best) = ransac(  //
            X, sara::NisterFivePointAlgorithm{},     //
            inlier_predicate, num_samples, data_normalizer, true);

        epipolar_edges.E_inliers[0] = inliers;
        epipolar_edges.E_best_samples[0] = sample_best;
      }

      auto& F = epipolar_edges.F[0];
      {
        F.matrix() = K_inv.transpose() * E.matrix() * K_inv;

        epipolar_edges.F_num_samples[0] = num_samples;
        epipolar_edges.F_noise = epipolar_edges.E_noise;
        epipolar_edges.F_inliers = epipolar_edges.E_inliers;
        epipolar_edges.F_best_samples = epipolar_edges.E_best_samples;
      }

      sara::print_stage("Draw...");
      for (auto m = 0u; m < matches.size(); ++m)
      {
        if (!inliers(m))
          continue;
        const auto& match = matches[m];
        sara::draw(match.x(), sara::Blue8);
        sara::draw(match.y(), sara::Cyan8);
        sara::draw_arrow(match.x_pos(), match.y_pos(), sara::Yellow8);
        sara::get_key();
      }
    }

    if (sara::get_key() == sara::KEY_ESCAPE)
      break;
  }

  return 0;
}
