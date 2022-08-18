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

#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/MultiViewGeometry/Calibration/OmnidirectionalCameraReprojectionError.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>

#include <drafts/Calibration/Chessboard.hpp>
#include <drafts/Calibration/HomographyDecomposition.hpp>
#include <drafts/Calibration/HomographyEstimation.hpp>


namespace sara = DO::Sara;


// This seems to work well...
static inline auto init_K(int w, int h) -> Eigen::Matrix3d
{
  const auto d = static_cast<double>(std::max(w, h));
  const auto f = 0.5 * d;

  // clang-format off
  const auto K = (Eigen::Matrix3d{} <<
    f, 0, w * 0.5,
    0, f, h * 0.5,
    0, 0,       1
  ).finished();
  // clang-format on

  return K;
}

inline auto inspect(sara::ImageView<sara::Rgb8>& image,                 //
                    const sara::ChessboardCorners& chessboard,          //
                    const sara::OmnidirectionalCamera<double>& camera,  //
                    const Eigen::Matrix3d& R,                           //
                    const Eigen::Vector3d& t,                           //
                    bool pause = false) -> void
{
  const auto o = chessboard.image_point(0, 0);
  const auto i = chessboard.image_point(1, 0);
  const auto j = chessboard.image_point(0, 1);
  const auto s = chessboard.square_size().value;


  const Eigen::Vector3d k3 = R * Eigen::Vector3d::UnitZ() * s + t;
  const Eigen::Vector2f k = camera.project(k3).cast<float>();

  static const auto red = sara::Rgb8{167, 0, 0};
  static const auto green = sara::Rgb8{89, 216, 26};
  sara::draw_arrow(image, o, i, red, 6);
  sara::draw_arrow(image, o, j, green, 6);
  sara::draw_arrow(image, o, k, sara::Blue8, 6);

  for (auto y = 0; y < chessboard.height(); ++y)
  {
    for (auto x = 0; x < chessboard.width(); ++x)
    {
      auto P = Eigen::Vector3d{};
      P << chessboard.scene_point(x, y).cast<double>(), 0;
      P = R * P + t;

      const Eigen::Vector2f p1 = chessboard.image_point(x, y);
      const Eigen::Vector2f p2 = camera.project(P).cast<float>();

      draw_circle(image, p1, 3.f, sara::Cyan8, 3);
      draw_circle(image, p2, 3.f, sara::Magenta8, 3);
      if (pause)
      {
        sara::display(image);
        sara::get_key();
      }
    }
  }
}


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

  inline auto set_camera_type(CameraType camera_type) -> void
  {
    _camera_type = camera_type;
  }

  inline auto initialize_intrinsics(const Eigen::Matrix3d& K,
                                    const Eigen::Vector3d& k,
                                    const Eigen::Vector2d& p,  //
                                    const double xi) -> void
  {
    // fx
    _intrinsics[0] = K(0, 0);
    // fy
    _intrinsics[1] = K(1, 1);
    // shear
    _intrinsics[2] = K(0, 1) / K(0, 0);
    // u0
    _intrinsics[3] = K(0, 2);
    // v0
    _intrinsics[4] = K(1, 2);
    // k
    _intrinsics[5] = k(0);
    _intrinsics[6] = k(1);
    // _intrinsics[7] = k(2);
    // p
    _intrinsics[7] = p(0);
    _intrinsics[8] = p(1);
    // xi
    _intrinsics[9] = xi;
  }

  inline auto add(const sara::ChessboardCorners& chessboard,
                  const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
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

  inline auto obs_2d() const -> const double*
  {
    return _observations_2d.data();
  }

  inline auto obs_3d() const -> const double*
  {
    return _observations_3d.data();
  }

  inline auto mutable_intrinsics() -> double*
  {
    return _intrinsics.data();
  }

  inline auto mutable_extrinsics(int n) -> double*
  {
    return _extrinsics.data() + extrinsic_parameter_count * n;
  }

  inline auto rotation(int n) const -> Eigen::AngleAxisd
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

  inline auto translation(int n) const -> Eigen::Vector3d
  {
    auto x = _extrinsics[extrinsic_parameter_count * n + 3 + 0];
    auto y = _extrinsics[extrinsic_parameter_count * n + 3 + 1];
    auto z = _extrinsics[extrinsic_parameter_count * n + 3 + 2];
    return {x, y, z};
  }

  inline auto transform_into_ceres_problem(ceres::Problem& problem) -> void
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

        auto cost_function =
            sara::OmnidirectionalCameraReprojectionError::create(image_point,
                                                                 scene_point);

        problem.AddResidualBlock(cost_function, loss_fn,  //
                                 mutable_intrinsics(), mutable_extrinsics(n));

        data_pos += 2;
      }
    }

    static constexpr auto fx = 0;
    static constexpr auto fy = 1;
    // alpha is the normalized_shear.
    static constexpr auto alpha = 2;  // shear = alpha * fx
    static constexpr auto k0 = 5;
    static constexpr auto k1 = 6;
    static constexpr auto p0 = 7;
    static constexpr auto p1 = 8;
    static constexpr auto xi = 9;

    // Bounds on the calibration matrix.
    for (const auto& f : {fx, fy})
    {
      problem.SetParameterLowerBound(mutable_intrinsics(), f, 500);
      problem.SetParameterUpperBound(mutable_intrinsics(), f, 5000);
    }
    // Normalized shear.
    // - We should not need any further adjustment on the bounds.
    problem.SetParameterLowerBound(mutable_intrinsics(), alpha, -1.);
    problem.SetParameterUpperBound(mutable_intrinsics(), alpha, 1.);
    // So far no need for (u0, v0)

    // Bounds on the distortion coefficients.
    // - We should not need any further adjustment on the bounds.
    for (const auto& dist_coeff : {k0, k1, p0, p1})
    {
      problem.SetParameterLowerBound(mutable_intrinsics(), dist_coeff, -1);
      problem.SetParameterUpperBound(mutable_intrinsics(), dist_coeff, 1);
    }

    // Bounds on the mirror parameter.
    //
    // - If we are dealing with a fisheye camera, we should freeze the xi
    //   parameter to 1.
    static constexpr auto tolerance =
        0.01;  // too little... and the optimizer may
               // get stuck into a bad local minimum.
    switch (_camera_type)
    {
    case CameraType::Fisheye:
      // - This is a quick and dirty approach...
      problem.SetParameterLowerBound(mutable_intrinsics(), xi, 1 - tolerance);
      problem.SetParameterUpperBound(mutable_intrinsics(), xi, 1 + tolerance);

      // Otherwise add a penalty residual block:
      // auto penalty_block = /* TODO */;
      // problem.AddResidualBlock(penalty_block, nullptr, mutable_intrinsics());
      break;
    case CameraType::Pinhole:
      problem.SetParameterLowerBound(mutable_intrinsics(), xi, -tolerance);
      problem.SetParameterUpperBound(mutable_intrinsics(), xi, tolerance);
      break;
    default:
      problem.SetParameterLowerBound(mutable_intrinsics(), xi, -10.);
      problem.SetParameterUpperBound(mutable_intrinsics(), xi, 10.);
      break;
    }
  }

  inline auto
  copy_camera_intrinsics(sara::OmnidirectionalCamera<double>& camera) -> void
  {
    // Copy back the final parameter to the omnidirectional camera parameter
    // object.
    const auto fx = mutable_intrinsics()[0];
    const auto fy = mutable_intrinsics()[1];
    const auto alpha = mutable_intrinsics()[2];
    const auto shear = fx * alpha;
    const auto u0 = mutable_intrinsics()[3];
    const auto v0 = mutable_intrinsics()[4];
    const auto k0 = mutable_intrinsics()[5];
    const auto k1 = mutable_intrinsics()[6];
    const auto p0 = mutable_intrinsics()[7];
    const auto p1 = mutable_intrinsics()[8];
    const auto xi = mutable_intrinsics()[9];
    // clang-format off
    auto K = Eigen::Matrix3d{};
    K << fx, shear, u0,
          0,   fy, v0,
          0,    0,  1;
    // clang-format on
    camera.set_calibration_matrix(K);
    camera.radial_distortion_coefficients << k0, k1, 0;
    camera.tangential_distortion_coefficients << p0, p1;
    camera.xi = xi;
  }

  inline auto reinitialize_extrinsic_parameters() -> void
  {
    throw std::runtime_error{"Implementation incomplete!"};

    const auto to_camera =
        [](const double*) -> sara::OmnidirectionalCamera<double> { return {}; };
    const auto camera = to_camera(_intrinsics.data());

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

#if 0
      const auto angle_axis = Eigen::AngleAxisd{Rs.front()};
      const auto& angle = angle_axis.angle();
      const auto& axis = angle_axis.axis();
#else
      for (auto k = 0; k < 3; ++k)
        angle_axis_ptr[k] = 0;  // angle * axis(k);

      for (auto k = 0; k < 3; ++k)
        translation_ptr[k] = 0;  // ts.front()(k);
#endif
    }
  }

private:
  int _w = 0;
  int _h = 0;
  int _num_images = 0;

  std::vector<int> _num_points_at_image;
  std::vector<double> _observations_2d;
  std::vector<double> _observations_3d;
  std::array<double, intrinsic_parameter_count> _intrinsics;
  std::vector<double> _extrinsics;
  CameraType _camera_type = CameraType::General;
};


int __main(int argc, char** argv)
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
        [](const auto& a, const auto& b) { return a.size(), b.size(); });
    return sara::ChessboardCorners{chessboard, square_size, pattern_size};
  };
  auto chessboards = std::vector<sara::ChessboardCorners>{};

  // Initialize the calibration matrix.
  auto camera = sara::OmnidirectionalCamera<double>{};
  camera.K = init_K(frame.width(), frame.height());
  camera.radial_distortion_coefficients.setZero();
  camera.tangential_distortion_coefficients.setZero();
  camera.xi = 1;

  // Initialize the calibration problem.
  auto calibration_problem = ChessboardCalibrationProblem{};
  calibration_problem.initialize_intrinsics(
      camera.K, camera.radial_distortion_coefficients,
      camera.tangential_distortion_coefficients, camera.xi);

  auto selected_frames = std::vector<sara::Image<sara::Rgb8>>{};
  sara::create_window(frame.sizes());
  sara::set_antialiasing();
  for (auto i = 0;; ++i)
  {
    if (!video_stream.read())
      break;

    if (i % 3 != 0)
      continue;

    if (selected_frames.size() > 40)
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

      auto Rs = std::vector<Eigen::Matrix3d>{};
      auto ts = std::vector<Eigen::Vector3d>{};
      auto ns = std::vector<Eigen::Vector3d>{};

#define USE_QR_FACTORIZATION
#ifdef USE_QR_FACTORIZATION
      // This simple approach gives the best results.
      const Eigen::Matrix3d H = estimate_H(chessboard).normalized();
      decompose_H_RQ_factorization(H, camera.K, Rs, ts, ns);
#else
      Rs = {Eigen::Matrix3d::Identity()};
      ts = {Eigen::Vector3d::Zero()};
      ns = {Eigen::Vector3d::Zero()};
#endif

      SARA_DEBUG << "\nR =\n" << Rs[0] << std::endl;
      SARA_DEBUG << "\nt =\n" << ts[0] << std::endl;
      SARA_DEBUG << "\nn =\n" << ns[0] << std::endl;

      calibration_problem.add(chessboard, Rs[0], ts[0]);

      inspect(frame_copy, chessboard, camera.K, Rs[0], ts[0]);
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

    SARA_DEBUG << "K =\n" << camera.K << std::endl;
    SARA_DEBUG << "k = " << camera.radial_distortion_coefficients.transpose()
               << std::endl;
    SARA_DEBUG << "p = "
               << camera.tangential_distortion_coefficients.transpose()
               << std::endl;
    SARA_DEBUG << "xi = " << camera.xi << std::endl;
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
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
