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

//! @example

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/DisjointSets/DisjointSets.hpp>
#include <DO/Sara/FeatureDetectors/EdgeDetector.hpp>
#include <DO/Sara/FeatureDetectors/EdgePostProcessing.hpp>
#include <DO/Sara/FeatureDetectors/EdgeUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/ImageProcessing/EdgeGrouping.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/CameraModel.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/BrownConradyDistortionModel.hpp>
#include <DO/Sara/MultiViewGeometry/SingleView/VanishingPoint.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace DO::Sara;


inline constexpr long double operator"" _percent(long double x)
{
  return x / 100;
}


auto initialize_camera_intrinsics_1()
{
  auto intrinsics = BrownConradyCamera32<float>{};

  const auto f = 991.8030424131325f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;
  intrinsics.image_sizes << 1920, 1080;
  // clang-format off
  intrinsics.set_calibration_matrix((Eigen::Matrix3f{} <<
    f, 0, u0,
    0, f, v0,
    0, 0,  1).finished());
  // clang-format on
  intrinsics.distortion_model.k.setZero();
  intrinsics.distortion_model.p.setZero();

  return intrinsics;
}

auto initialize_camera_intrinsics_2()
{
  auto intrinsics = BrownConradyCamera32<float>{};

  const auto f = 946.8984425572634f;
  const auto u0 = 960.f;
  const auto v0 = 540.f;
  intrinsics.image_sizes << 1920, 1080;
  // clang-format off
  intrinsics.set_calibration_matrix((Eigen::Matrix3f{} <<
    f, 0, u0,
    0, f, v0,
    0, 0,  1).finished());
  intrinsics.distortion_model.k <<
    -0.22996356451342749f,
    0.05952465745165465f,
    -0.007399008111054717f;
  // clang-format on
  intrinsics.distortion_model.p.setZero();

  return intrinsics;
}


auto initialize_no_crop_region(const Eigen::Vector2i& sizes)
{
  const Eigen::Vector2i& p1 = Eigen::Vector2i::Zero();
  const Eigen::Vector2i& p2 = sizes;
  return std::make_pair(p1, p2);
}

auto initialize_crop_region_1(const Eigen::Vector2i& sizes)
{
  const Eigen::Vector2i& p1 = {0, 0.6 * sizes.y()};
  const Eigen::Vector2i& p2 = {sizes.x(), 0.9 * sizes.y()};
  return std::make_pair(p1, p2);
}

auto initialize_crop_region_2(const Eigen::Vector2i& sizes)
{
  const Eigen::Vector2i& p1 = {0.1 * sizes.x(), 0.2 * sizes.y()};
  const Eigen::Vector2i& p2 = {0.9 * sizes.x(), 0.75 * sizes.y()};
  return std::make_pair(p1, p2);
}


auto default_camera_matrix()
{
  auto P = Eigen::Matrix<float, 3, 4>{};
  P.setZero();
  P.block<3, 3>(0, 0).setIdentity();
  return P;
}

auto match_rotational_axes(const Eigen::Matrix3f& R0,    //
                           const Eigen::Matrix3f& dirs,  //
                           Eigen::Matrix3f& R1)
{
  // The best matching axes.
  struct Match
  {
    int j0;
    int j1;
  };
  auto match_scores = std::array<std::pair<float, Match>, 3>{};

  // Find the axes that aligns best.
  const Eigen::Matrix3f dots = (R0.transpose() * dirs).cwiseAbs();
  SARA_DEBUG << "dots =\n" << dots << std::endl;

  for (auto i = 0; i < 3; ++i)
  {
    auto& [score, match] = match_scores[i];
    match.j0 = i;
    score = dots.row(i).maxCoeff(&match.j1);
  }

  // Best matching axis among the three matches.
  std::sort(match_scores.begin(), match_scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }  //
  );

  // Flimsy algorithm.
  for (const auto& [score, match] : match_scores)
  {
    const auto& [j0, j1] = match;

    R1.col(j0) = dirs.col(j1);

    // Flip the sign if necessary.
    const auto cosine = R0.col(j0).dot(R1.col(j0));
    if (cosine < 0)
      R1.col(j0) *= -1;

    SARA_DEBUG << "Matching: " << j0 << " to  " << j1 << std::endl;
    SARA_DEBUG << "Matching score: " << score << std::endl;
    SARA_DEBUG << "  R0.col(" << j0 << ") = " << R0.col(j0).transpose()
               << std::endl;
    SARA_DEBUG << "  R1.col(" << j0 << ") = " << R1.col(j0).transpose()
               << std::endl;
  }

  SARA_DEBUG << "R1 = \n" << R1 << std::endl;
}

auto calculate_yaw_and_pitch(const Eigen::Vector3f& v)
{
  return std::make_pair(std::atan2(v.x(), v.z()), std::asin(-v.y()));
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}


int sara_graphics_main(int argc, char** argv)
{
  using namespace std::string_literals;

  auto video_filepath = std::string{};
  auto downscale_factor = int{};
  auto skip = int{};

  po::options_description desc("Orthogonal Vanishing Point Detector");
  desc.add_options()
      ("help", "Usage")
      ("video,v", po::value<std::string>(&video_filepath),
       "input video file")
      ("downscale-factor,d",
       po::value<int>(&downscale_factor)->default_value(2),
       "downscale factor")
      ("skip,s", po::value<int>(&skip)->default_value(0),
       "number of frames to skip")
      ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return 1;
  }

  if (!vm.count("video"))
  {
    std::cout << "The video file must be specified!\n" << desc << "\n";
    return 1;
  }

  // OpenMP.
#ifdef _OPENMP
  omp_set_num_threads(omp_get_max_threads());
#endif

  // Input and output from Sara.
  VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();
  auto frame_undistorted = Image<Rgb8>{video_stream.sizes()};
  auto frame_gray32f = Image<float>{video_stream.sizes()};
  auto screen_contents = Image<Rgb8>{video_stream.sizes()};


  // Output save.
  const auto basename = fs::basename(video_filepath);
  VideoWriter video_writer{
#ifdef __APPLE__
      "/Users/david/Desktop/" + basename + ".ortho-vp.mp4",
#else
      "/home/david/Desktop/" + basename + ".ortho-vp.mp4",
#endif
      frame.sizes()
  };


  // Show the local extrema.
  create_window(frame.sizes());
  set_antialiasing();

  constexpr float high_threshold_ratio = static_cast<float>(20._percent);
  constexpr float low_threshold_ratio =
      static_cast<float>(high_threshold_ratio / 2.);
  constexpr float angular_threshold = static_cast<float>((20._deg).value);
  const auto sigma = std::sqrt(std::pow(1.2f, 2) - 1);
  // const auto [p1, p2] = initialize_no_crop_region(frame.sizes());
  const auto [p1, p2] = initialize_crop_region_2(frame.sizes());

  auto ed = EdgeDetector{{
      high_threshold_ratio,
      low_threshold_ratio,
      angular_threshold
  }};


  // Initialize the camera matrix.
  const auto intrinsics = initialize_camera_intrinsics_2();
  auto P = default_camera_matrix();
  P = intrinsics.K * P;
  const auto Pt = P.transpose().eval();

  auto R0 = Eigen::Matrix3f{};
  auto R1 = Eigen::Matrix3f{};
  R0.setZero();
  R1.setZero();

  auto frames_read = 0;
  while (true)
  {
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;
    SARA_DEBUG << "Processing frame " << frames_read << std::endl;

    tic();
    undistort(intrinsics, frame, frame_undistorted);
    toc("Undistort");

    // Reduce our attention to the central part of the image.
    tic();
    const auto frame_cropped = crop(frame_undistorted, p1, p2);
    toc("Crop");

    tic();
    frame_gray32f = frame_cropped.convert<float>();
    toc("Grayscale");

    tic();
    frame_gray32f = gaussian(frame_gray32f, sigma);
    toc("Blur");

    if (downscale_factor > 1)
    {
      tic();
      frame_gray32f = downscale(frame_gray32f, downscale_factor);
      toc("Downscale");
    }

    ed(frame_gray32f);
    auto& edges_refined = ed.pipeline.edges_simplified;

    tic();
    // TODO: split only if the inertias matrix is becoming isotropic.
    edges_refined = split(edges_refined, 10. * M_PI / 180.);
    toc("Edge Split");

    tic();
    auto edges = std::vector<std::vector<Eigen::Vector2d>>{};
    edges.reserve(edges_refined.size());
    for (const auto& e : edges_refined)
    {
      if (e.size() < 2)
        continue;
      if (length(e) < 20)
        continue;
      edges.emplace_back(e);
    }
    toc("Edge Filtering");

    tic();
    const auto edge_stats = CurveStatistics{edges};
    toc("Edge Shape Statistics");

    tic();
    auto line_segments = extract_line_segments_quick_and_dirty(edge_stats);
    SARA_CHECK(line_segments.size());


    // Go back to the original pixel coordinates for single-view geometry
    // analysis.
    const Eigen::Vector2d p1d = p1.cast<double>();
    const auto s = static_cast<float>(downscale_factor);
    for (auto& ls : line_segments)
    {
      ls.p1() = p1d + s * ls.p1();
      ls.p2() = p1d + s * ls.p2();
    }
    const auto lines = to_lines(line_segments);
    toc("Line Segment Extraction");

    tic();
    const Eigen::MatrixXf lines_as_matrix = lines.matrix().transpose();
    const Eigen::MatrixXf planes_backprojected = (Pt * lines_as_matrix)  //
                                                     .colwise()          //
                                                     .normalized();
    toc("Planes Backprojected");

    tic();
    const auto planes_tensor = TensorView_<float, 2>{
        const_cast<float*>(planes_backprojected.data()),
        {planes_backprojected.cols(), planes_backprojected.rows()}};

    const auto angle_threshold = std::sin(float((3._deg).value));
    SARA_DEBUG << "planes_tensor.rows = " << planes_tensor.rows() << std::endl;
    const auto ransac_result = find_dominant_orthogonal_directions(  //
        planes_tensor,                                               //
        angle_threshold,                                             //
        100);
    const auto dirs = std::get<0>(ransac_result);
    const auto inliers = std::get<1>(ransac_result);
    toc("Vanishing Point");

    const auto inliers_count = inliers.flat_array().count();

    tic();
    if (inliers_count > 0)
    {
      if (R1.squaredNorm() < 1e-5f)
        R1 = dirs;
      else
      {
        // Save the previous rotation matrix.
        R0 = R1;
        match_rotational_axes(R0, dirs, R1);
      }
    }
    toc("Axis Matching");

    auto& detection = frame_undistorted;
#ifdef CLEAR_IMAGE
    detection.flat_array().fill(Black8);
#endif

    tic();
    {
      if (inliers_count > 0)
      {
        SARA_DEBUG << "inliers = " << inliers.flat_array().count() << std::endl;
        SARA_DEBUG << "R =\n" << R1 << std::endl;
        SARA_DEBUG << "|R| = " << R1.determinant() << std::endl;

        for (auto i = 0u; i < line_segments.size(); ++i)
        {
          if (!inliers(i))
            continue;

          const auto& ls = line_segments[i];
          const auto& a = ls.p1();
          const auto& b = ls.p2();

          const Eigen::Vector3f n = planes_backprojected.col(i).head(3);
          const auto rn = std::array<float, 3>{std::abs(R1.col(0).dot(n)),
                                               std::abs(R1.col(1).dot(n)),
                                               std::abs(R1.col(2).dot(n))};
          const auto ibest =
              std::min_element(rn.begin(), rn.end()) - rn.begin();

          static constexpr auto int_round = [](const double x) {
            return static_cast<int>(std::round(x));
          };
          if (ibest == 0)
            draw_line(detection, int_round(a.x()), int_round(a.y()),
                      int_round(b.x()), int_round(b.y()), Red8, 4);
          else if (ibest == 1)
            draw_line(detection, int_round(a.x()), int_round(a.y()),
                      int_round(b.x()), int_round(b.y()), Green8, 4);
          else
            draw_line(detection, int_round(a.x()), int_round(a.y()),
                      int_round(b.x()), int_round(b.y()), Blue8, 4);
        }
      }

      display(detection);
      if (inliers_count > 0 && std::abs(R0.determinant()) > 0.9f)
      {
        const auto angular_diff_degree =
            angular_distance(R0, R1) / M_PI * 180.f;
        draw_text(100, 100,
                  format("Angle diff = %0.2f degree", angular_diff_degree),
                  White8, 20, 0, false, true);
      }
    }
    toc("Display");

    tic();
    grab_screen_contents(screen_contents);
    video_writer.write(screen_contents);
    toc("Video Write");
  }

  return 0;
}
