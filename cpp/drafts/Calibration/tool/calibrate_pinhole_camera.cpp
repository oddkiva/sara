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

#include <drafts/Calibration/Utilities.hpp>

#include <DO/Sara/Core/EigenFormatInterop.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/MultiViewGeometry/Calibration/PinholeCameraReprojectionError.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/v2/PinholeCamera.hpp>
#include <DO/Sara/MultiViewGeometry/PnP/LambdaTwist.hpp>


namespace sara = DO::Sara;


class ChessboardCalibrationData
{
public:
  static constexpr auto extrinsic_parameter_count =
      sara::PinholeCameraReprojectionError::extrinsic_parameter_count;

  auto initialize_intrinsics(const sara::v2::PinholeCamera<double>& camera) -> void
  {
    // fx
    _intrinsics.fx() = camera.fx();
    // fy (NORMALIZED)
    _intrinsics.fy() = camera.fy() / camera.fx();
    // shear (NORMALIZED)
    _intrinsics.shear() = camera.shear() / camera.fx();
    // u0
    _intrinsics.u0() = camera.u0();
    // v0
    _intrinsics.v0() = camera.v0();
  }

  auto add(const sara::ChessboardCorners& chessboard,  //
           const Eigen::Matrix3d& R, const Eigen::Vector3d& t) -> void
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
    return _intrinsics.data.data();
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
    auto data_pos = std::size_t{};
    for (auto n = 0; n < _num_images; ++n)
    {
      const auto& num_points = _num_points_at_image[n];
      for (auto p = 0; p < num_points; ++p)
      {
        const Eigen::Vector2d image_point = Eigen::Map<const Eigen::Vector2d>(
            _observations_2d.data() + data_pos);
        const Eigen::Vector2d scene_point = Eigen::Map<const Eigen::Vector2d>(
            _observations_3d.data() + data_pos);

        auto cost_function = sara::PinholeCameraReprojectionError::create(
            image_point, scene_point);

        problem.AddResidualBlock(cost_function, loss_fn,                //
                                 &_intrinsics.fx(), &_intrinsics.fy(),  //
                                 &_intrinsics.shear(),                  //
                                 _intrinsics.principal_point().data(),  //
                                 mutable_extrinsics(n));

        data_pos += 2;
      }
    }

    // Bounds on fx.
    problem.SetParameterLowerBound(&_intrinsics.fx(), 0, 500);
    problem.SetParameterUpperBound(&_intrinsics.fx(), 0, 5000);

    // Bounds on fy (NORMALIZED).
    problem.SetParameterLowerBound(&_intrinsics.fy(), 0, 0.1);
    problem.SetParameterUpperBound(&_intrinsics.fy(), 0, 2.0);

    // Bounds on the shear (NORMALIZED).
    // - We should not need any further adjustment on the bounds.
    problem.SetParameterLowerBound(&_intrinsics.shear(), 0, -1.);
    problem.SetParameterUpperBound(&_intrinsics.shear(), 0, 1.);

    // So far no need for (u0, v0)
  }

  auto fix_fy_normalized(ceres::Problem& problem) -> void
  {
    problem.SetParameterBlockConstant(&_intrinsics.fy());
  }

  auto fix_shear_normalized(ceres::Problem& problem) -> void
  {
    problem.SetParameterBlockConstant(&_intrinsics.shear());
  }

  auto fix_principal_point(ceres::Problem& problem) -> void
  {
    problem.SetParameterBlockConstant(_intrinsics.principal_point().data());
  }

  auto copy_camera_intrinsics(sara::v2::PinholeCamera<double>& camera) -> void
  {
    // Copy back the final parameter to the omnidirectional camera parameter
    // object.
    const auto fx = _intrinsics.fx();
    const auto fy_normalized = _intrinsics.fy();
    const auto fy = fx * fy_normalized;
    const auto shear_normalized = _intrinsics.shear();
    const auto shear = fx * shear_normalized;
    const auto u0 = _intrinsics.u0();
    const auto v0 = _intrinsics.v0();
    camera.fx() = fx;
    camera.fy() = fy;
    camera.shear() = shear;
    camera.u0() = u0;
    camera.v0() = v0;
  }

  auto reinitialize_extrinsic_parameters() -> void
  {
    throw std::runtime_error{"Implementation incomplete!"};

    const auto camera = sara::v2::PinholeCamera<double>{};

    auto data_pos = std::size_t{};

    for (auto image_index = 0; image_index < _num_images; ++image_index)
    {
      const auto& num_points = _num_points_at_image[image_index];
      auto p2ns = Eigen::MatrixXd{3, num_points};
      auto p3s = Eigen::MatrixXd{3, num_points};

      auto c = 0;
      for (auto y = 0; y < _h; ++y)
      {
        for (auto x = 0; x < _w; ++x)
        {
          const Eigen::Vector2d p2 = Eigen::Map<const Eigen::Vector2d>(
              _observations_2d.data() + data_pos);

          const Eigen::Vector3d p3 = Eigen::Map<const Eigen::Vector2d>(
                                         _observations_2d.data() + data_pos)
                                         .homogeneous();

          const Eigen::Vector3d ray = camera.backproject(p2);
          const Eigen::Vector2d p2n = ray.hnormalized();

          p2ns.col(c) << p2n, 1;
          p3s.col(c) = p3;

          ++c;
          data_pos += 2;
        }
      }

      // TODO: reestimate the extrinsics with the PnP algorithm.
      auto extrinsics = mutable_extrinsics(image_index);
      auto angle_axis_ptr = extrinsics;
      auto translation_ptr = extrinsics + 3;

      for (auto k = 0; k < 3; ++k)
        angle_axis_ptr[k] = 0;

      for (auto k = 0; k < 3; ++k)
        translation_ptr[k] = 0;
    }
  }

private:
  int _w = 0;
  int _h = 0;
  int _num_images = 0;

  std::vector<int> _num_points_at_image;
  std::vector<double> _observations_2d;
  std::vector<double> _observations_3d;
  sara::v2::PinholeCamera<double> _intrinsics;
  std::vector<double> _extrinsics;
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
  auto camera = sara::v2::PinholeCamera<double>{};
  sara::init_calibration_matrix(camera, frame.width(), frame.height());

  // Initialize the calibration problem.
  auto calibration_data = ChessboardCalibrationData{};
  calibration_data.initialize_intrinsics(camera);

  auto selected_frames = std::vector<sara::Image<sara::Rgb8>>{};
  sara::create_window(frame.sizes());
  sara::set_antialiasing();
  for (auto i = 0;; ++i)
  {
    if (!video_stream.read())
      break;

    if (i % 3 != 0)
      continue;

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

      calibration_data.add(chessboard, R, t);

      // inspect(frame_copy, chessboard, K_initial, Rs[0], ts[0]);
      inspect(frame_copy, chessboard, camera, R, t);
      sara::draw_text(frame_copy, 80, 80, "Chessboard: FOUND!", sara::White8,
                      60, 0, false, true);
      sara::display(frame_copy);

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
  calibration_data.transform_into_ceres_problem(problem);
  calibration_data.fix_fy_normalized(problem);
  calibration_data.fix_shear_normalized(problem);
  calibration_data.fix_principal_point(problem);

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

    calibration_data.copy_camera_intrinsics(camera);

    SARA_DEBUG << fmt::format("camera.fx() = {}\n", camera.fx());
    SARA_DEBUG << fmt::format("camera.fy() = {}\n", camera.fy());
    SARA_DEBUG << fmt::format("camera.shear() = {}\n", camera.shear());
    SARA_DEBUG << fmt::format("camera.principal_point() = {}\n",
                              camera.principal_point().transpose().eval());
  }

  for (auto i = 0u; i < chessboards.size(); ++i)
  {
    const auto R = calibration_data.rotation(i).toRotationMatrix();
    const auto t = calibration_data.translation(i);

    auto frame_copy = selected_frames[i];
    inspect(frame_copy, chessboards[i], camera, R, t);
    sara::display(frame_copy);
    sara::get_key();
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(sara_graphics_main);
  return app.exec();
}
