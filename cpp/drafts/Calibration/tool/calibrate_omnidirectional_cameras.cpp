// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <drafts/Calibration/Chessboard.hpp>
#include <drafts/Calibration/Utilities.hpp>

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/MultiViewGeometry/Calibration/OmnidirectionalCameraReprojectionError.hpp>


namespace sara = DO::Sara;


class ChessboardCalibrationProblem
{
public:
  static constexpr auto intrinsic_parameter_count =
      sara::OmnidirectionalCameraReprojectionError::intrinsic_parameter_count;
  static constexpr auto extrinsic_parameter_count =
      sara::OmnidirectionalCameraReprojectionError::extrinsic_parameter_count;

  enum CameraType : std::uint8_t
  {
    Pinhole,
    Fisheye,
    General
  };

  auto set_camera_type(CameraType camera_type) -> void
  {
    _camera_type = camera_type;
  }

  auto
  initialize_intrinsics(const sara::v2::OmnidirectionalCamera<double>& camera)
      -> void
  {
    // fx
    _intrinsics[0] = camera.fx();
    // fy (NORMALIZED)
    _intrinsics[1] = camera.fy() / camera.fx();
    // shear (NORMALIZED)
    _intrinsics[2] = camera.shear() / camera.fx();
    // principal point (u0, v0)
    _intrinsics[3] = camera.u0();
    _intrinsics[4] = camera.v0();

    // Mirror parameter: xi
    _intrinsics[5] = camera.xi();

    // k
    _intrinsics[6] = camera.k()(0);
    _intrinsics[7] = camera.k()(1);
    _intrinsics[8] = camera.k()(2);
    // p
    _intrinsics[9] = camera.p()(0);
    _intrinsics[10] = camera.p()(1);
  }

  auto add(const sara::ChessboardCorners& chessboard, const Eigen::Matrix3d& R,
           const Eigen::Vector3d& t)
  {
    ++_num_images;

    // Initialize the rotation.
    const auto angle_axis = Eigen::AngleAxisd{R};
    const auto& angle = angle_axis.angle();
    const auto& axis = angle_axis.axis();
    for (auto i = 0; i < 3; ++i)
      _extrinsics.push_back(angle * axis(i));

    // Initialize the translation.
    for (auto i = 0; i < 3; ++i)
      _extrinsics.push_back(t(i));

    // Store the image points.
    auto num_points = 0;
    const auto& square_size = chessboard.square_size().value;
    for (auto y = 0; y < chessboard.height(); ++y)
    {
      for (auto x = 0; x < chessboard.width(); ++x)
      {
        const Eigen::Vector2f image_point = chessboard.image_point(x, y);
        if (sara::is_nan(image_point))
          continue;
        _observations_2d.push_back(image_point.x());
        _observations_2d.push_back(image_point.y());

        _observations_3d.push_back(x * square_size);
        _observations_3d.push_back(y * square_size);
        _observations_3d.push_back(0);

        ++num_points;
      }
    }

    _num_points_at_image.push_back(num_points);
  }

  auto obs_2d() const -> const double*
  {
    return _observations_2d.data();
  }

  auto obs_3d() const -> const double*
  {
    return _observations_3d.data();
  }

  auto mutable_intrinsics() -> double*
  {
    return _intrinsics.data();
  }

  auto mutable_extrinsics(int n) -> double*
  {
    return _extrinsics.data() + extrinsic_parameter_count * n;
  }

  auto rotation(int n) const -> Eigen::AngleAxisd
  {
    auto x = _extrinsics[extrinsic_parameter_count * n + 0];
    auto y = _extrinsics[extrinsic_parameter_count * n + 1];
    auto z = _extrinsics[extrinsic_parameter_count * n + 2];
    const auto angle = std::sqrt(x * x + y * y + z * z);
    x /= angle;
    y /= angle;
    z /= angle;

    return {angle, Eigen::Vector3d{x, y, z}};
  }

  auto translation(int n) const -> Eigen::Vector3d
  {
    auto x = _extrinsics[extrinsic_parameter_count * n + 3 + 0];
    auto y = _extrinsics[extrinsic_parameter_count * n + 3 + 1];
    auto z = _extrinsics[extrinsic_parameter_count * n + 3 + 2];
    return {x, y, z};
  }

  auto transform_into_ceres_problem(ceres::Problem& problem) -> void
  {
    auto loss_fn = nullptr;  // new ceres::HuberLoss{1.0};
    auto obs_ptr = _observations_2d.data();
    auto scene_ptr = _observations_3d.data();
    for (auto n = 0; n < _num_images; ++n)
    {
      const auto& num_points = _num_points_at_image[n];
      for (auto p = 0; p < num_points; ++p)
      {
        const Eigen::Vector2d image_point =
            Eigen::Map<const Eigen::Vector2d>(obs_ptr);
        const Eigen::Vector3d scene_point =
            Eigen::Map<const Eigen::Vector3d>(scene_ptr);

        auto cost_function =
            sara::OmnidirectionalCameraReprojectionError::create(image_point,
                                                                 scene_point);

        problem.AddResidualBlock(cost_function, loss_fn,  //
                                 mutable_intrinsics(), mutable_extrinsics(n));

        obs_ptr += 2;
        scene_ptr += 3;
      }
    }

    static constexpr auto fx = 0;
    static constexpr auto fy_normalized = 1;
    // alpha is the normalized_shear.
    static constexpr auto alpha = 2;  // shear = alpha * fx
    [[maybe_unused]] static constexpr auto u0 = 3;
    [[maybe_unused]] static constexpr auto v0 = 4;
    static constexpr auto xi = 5;
    static constexpr auto k0 = 6;
    static constexpr auto k1 = 7;
    static constexpr auto k2 = 8;
    static constexpr auto p0 = 9;
    static constexpr auto p1 = 10;

    // Bounds on the calibration matrix.
    problem.SetParameterLowerBound(mutable_intrinsics(), fx, 500);
    problem.SetParameterUpperBound(mutable_intrinsics(), fx, 5000);

    problem.SetParameterLowerBound(mutable_intrinsics(), fy_normalized, 0.);
    problem.SetParameterUpperBound(mutable_intrinsics(), fy_normalized, 2.);

    // Normalized shear.
    // - We should not need any further adjustment on the bounds.
    problem.SetParameterLowerBound(mutable_intrinsics(), alpha, -1.);
    problem.SetParameterUpperBound(mutable_intrinsics(), alpha, 1.);
    // So far no need for (u0, v0)

    // Bounds on the distortion coefficients.
    // - We should not need any further adjustment on the bounds.
    for (const auto& idx : {k0, k1, k2, p0, p1})
    {
      problem.SetParameterLowerBound(mutable_intrinsics(), idx, -1.);
      problem.SetParameterUpperBound(mutable_intrinsics(), idx, 1.);
    }

    // for (const auto& i : {u0, v0})
    // {
    //   problem.SetParameterLowerBound(mutable_intrinsics(), i,
    //   mutable_intrinsics()[i] - 1);
    //   problem.SetParameterUpperBound(mutable_intrinsics(), i,
    //   mutable_intrinsics()[i] + 1);
    // }

    // Bounds on the mirror parameter.
    //
    // - If we are dealing with a fisheye camera, we should freeze the xi
    //   parameter to 1.
    //
    // By default freeze the principal point.
    auto intrinsics_to_freeze = std::vector<int>{};
    switch (_camera_type)
    {
    case CameraType::Fisheye:
      mutable_intrinsics()[xi] = 1.;
      intrinsics_to_freeze.push_back(xi);
      break;
    case CameraType::Pinhole:
      mutable_intrinsics()[xi] = 0.;
      intrinsics_to_freeze.push_back(xi);
      break;
    default:
      problem.SetParameterLowerBound(mutable_intrinsics(), xi, -10.);
      problem.SetParameterUpperBound(mutable_intrinsics(), xi, 10.);
      break;
    }

    // Impose a fixed principal point.
#if CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2
    auto intrinsic_manifold = new ceres::SubsetManifold{
        intrinsic_parameter_count, intrinsics_to_freeze};
    problem.SetManifold(mutable_intrinsics(), intrinsic_manifold);
#else
    auto intrinsic_manifold = new ceres::SubsetParameterization{
        intrinsic_parameter_count, intrinsics_to_freeze};
    problem.SetParameterization(mutable_intrinsics(), intrinsic_manifold);
#endif
  }

  auto copy_camera_intrinsics(sara::v2::OmnidirectionalCamera<double>& camera)
      -> void
  {
    // Copy back the final parameter to the omnidirectional camera parameter
    // object.

    // Calibration matrix.
    const auto& fx = mutable_intrinsics()[0];
    const auto& fy_normalized = mutable_intrinsics()[1];
    const auto& alpha = mutable_intrinsics()[2];
    const auto fy = fy_normalized * fx;
    const auto shear = fx * alpha;
    const auto& u0 = mutable_intrinsics()[3];
    const auto& v0 = mutable_intrinsics()[4];
    camera.fx() = fx;
    camera.fy() = fy;
    camera.shear() = shear;
    camera.u0() = u0;
    camera.v0() = v0;

    // Mirror parameter.
    const auto& xi = mutable_intrinsics()[5];

    // Distortion parameters.
    const auto& k0 = mutable_intrinsics()[6];
    const auto& k1 = mutable_intrinsics()[7];
    const auto& k2 = mutable_intrinsics()[8];
    const auto& p0 = mutable_intrinsics()[9];
    const auto& p1 = mutable_intrinsics()[10];
    camera.k() << k0, k1, k2;
    camera.p() << p0, p1;
    camera.xi() = xi;
  }

private:
  int _num_images = 0;

  std::vector<int> _num_points_at_image;
  std::vector<double> _observations_2d;
  std::vector<double> _observations_3d;
  std::array<double, intrinsic_parameter_count> _intrinsics;
  std::vector<double> _extrinsics;
  CameraType _camera_type = CameraType::Pinhole;
};


auto sara_graphics_main(int argc, char** argv) -> int
{
  if (argc < 5)
  {
    std::cerr << "Usage: " << argv[0]
              << " VIDEO_PATH CHESSBOARD_SIZES CHESSBOARD_SQUARE_SIZE_IN_METERS"
              << std::endl;
    return 1;
  }

  const auto video_filepath = argv[1];
  const auto pattern_size = Eigen::Vector2i{
      std::stoi(argv[2]), std::stoi(argv[3])  //
  };
  const auto square_size = sara::Length{std::stod(argv[4])};

  auto video_stream = sara::VideoStream{video_filepath};
  auto frame = video_stream.frame();
  auto frame_gray32f = sara::Image<float>{frame.sizes()};

  auto chessboard_detector = sara::ChessboardDetector{};
  // Initialize the chessboard detector.
  {
    chessboard_detector.initialize_multiscale_harris_corner_detection_params(
        false, 3);
    chessboard_detector.initialize_filter_radius_according_to_scale();
    chessboard_detector.initialize_edge_detector();
  }
  const auto detect =
      [&chessboard_detector, &square_size, &pattern_size](
          const sara::ImageView<float>& frame) -> sara::ChessboardCorners {
    const auto& chessboards = chessboard_detector(frame);
    if (chessboards.empty())
      return sara::ChessboardCorners{{}, square_size, pattern_size};
    const auto& chessboard = *std::max_element(
        chessboards.rbegin(), chessboards.rend(),
        [](const auto& a, const auto& b) { return a.size() < b.size(); });
    return sara::ChessboardCorners{chessboard, square_size, pattern_size};
  };
  auto chessboards = std::vector<sara::ChessboardCorners>{};

  // Initialize the calibration matrix.
  auto camera = sara::v2::OmnidirectionalCamera<double>{};
  sara::init_calibration_matrix(camera, frame.width(), frame.height());
  camera.k().setZero();
  camera.p().setZero();
  camera.xi() = 1;

  // Initialize the calibration problem.
  auto calibration_problem = ChessboardCalibrationProblem{};
  calibration_problem.initialize_intrinsics(camera);

  auto selected_frames = std::vector<sara::Image<sara::Rgb8>>{};
  sara::create_window(frame.sizes());
  sara::set_antialiasing();
  for (auto i = 0;; ++i)
  {
    if (!video_stream.read())
      break;

    if (i % 3 != 0)
      continue;

    if (selected_frames.size() > 100)
      break;

    sara::tic();
    sara::from_rgb8_to_gray32f(frame, frame_gray32f);
    const auto chessboard = detect(frame_gray32f);
    sara::toc("Chessboard corner detection");

    const auto total_count = pattern_size.x() * pattern_size.y();
    const auto minimum_count = total_count / 3;
    if (chessboard.corner_count() > minimum_count)
    {
      SARA_DEBUG << "Chessboard found!\n";

      auto frame_copy = sara::Image<sara::Rgb8>{frame};
      draw_chessboard(frame_copy, chessboard);

      const auto pose = estimate_pose_with_p3p(chessboard, camera);
      if (pose == std::nullopt || sara::is_nan(*pose))
        continue;

      const Eigen::Matrix3d R = pose->topLeftCorner<3, 3>();
      const Eigen::Vector3d t = pose->col(3);
      SARA_DEBUG << "\nR =\n" << R << std::endl;
      SARA_DEBUG << "\nt =\n" << t << std::endl;

      calibration_problem.add(chessboard, R, t);

      inspect(frame_copy, chessboard, camera, R, t);
      sara::display(frame_copy);
      sara::draw_text(frame_copy, 80, 80, "Chessboard: FOUND!", sara::White8,
                      60, 0, false, true);

      selected_frames.emplace_back(video_stream.frame());
      chessboards.emplace_back(std::move(chessboard));
    }
    else
    {
      sara::display(frame);
      sara::draw_text(80, 80, "Chessboard: NOT FOUND!", sara::White8, 60, 0,
                      false, true);
      SARA_DEBUG << "[" << i << "] No chessboard found!" << std::endl;
    }
  }

  SARA_DEBUG << "Instantiating Ceres Problem..." << std::endl;
  auto problem = ceres::Problem{};
  calibration_problem.set_camera_type(
      ChessboardCalibrationProblem::CameraType::General);
  calibration_problem.transform_into_ceres_problem(problem);

  // Restarting the optimization solver is better than increasing the number of
  // iterations.
  //
  // cf. https://groups.google.com/g/ceres-solver/c/SEtXIMQwq88
  // Quoting Sameer Agarwal:
  // "my guess is that this is because the LBFGS direction is poor and
  // re-starting the solver resets it to an identity matrix."
  auto convergence = false;
  for (auto i = 0; i < 20 && !convergence; ++i)
  {
    SARA_DEBUG << "Solving Ceres Problem..." << std::endl;
    auto solver_options = ceres::Solver::Options{};
    solver_options.max_num_iterations = 500;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.update_state_every_iteration = true;
    solver_options.minimizer_progress_to_stdout = true;
    auto summary = ceres::Solver::Summary{};
    ceres::Solve(solver_options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    const auto rms_init =
        std::sqrt(summary.initial_cost / summary.num_residuals);
    const auto rms_final =
        std::sqrt(summary.final_cost / summary.num_residuals);
    SARA_DEBUG << "RMS[INITIAL] = " << rms_init << std::endl;
    SARA_DEBUG << "RMS[FINAL  ] = " << rms_final << std::endl;

    if (summary.termination_type == ceres::CONVERGENCE && rms_final < 1.)
      convergence = true;

    calibration_problem.copy_camera_intrinsics(camera);

    SARA_DEBUG << "fx = " << camera.fx() << std::endl;
    SARA_DEBUG << "fy = " << camera.fy() << std::endl;
    SARA_DEBUG << "shear = " << camera.shear() << std::endl;
    SARA_DEBUG << "u0 = " << camera.u0() << std::endl;
    SARA_DEBUG << "v0 = " << camera.v0() << std::endl;
    SARA_DEBUG << "k = " << camera.k().transpose() << std::endl;
    SARA_DEBUG << "p = " << camera.p().transpose() << std::endl;
    SARA_DEBUG << "xi = " << camera.xi() << std::endl;
  }

  for (auto i = 0u; i < chessboards.size(); ++i)
  {
    const auto R = calibration_problem.rotation(i).toRotationMatrix();
    const auto t = calibration_problem.translation(i);

    auto frame_copy = selected_frames[i];
    inspect(frame_copy, chessboards[i], camera, R, t);
    sara::display(frame_copy);
    sara::get_key();
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  auto app = sara::GraphicsApplication{argc, argv};
  app.register_user_main(sara_graphics_main);
  return app.exec();
}
