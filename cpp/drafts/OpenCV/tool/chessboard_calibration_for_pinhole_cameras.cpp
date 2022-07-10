#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/Calibration/PinholeCameraReprojectionError.hpp>
#include <DO/Sara/MultiViewGeometry/Resectioning/HartleyZisserman.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/OpenCV/HomographyDecomposition.hpp>
#include <drafts/OpenCV/HomographyEstimation.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


namespace sara = DO::Sara;

using sara::operator""_cm;


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


class ChessboardCalibrationProblem
{
public:
  static constexpr auto intrinsic_parameter_count =
      sara::PinholeCameraReprojectionError::intrinsic_parameter_count;
  static constexpr auto extrinsic_parameter_count =
      sara::PinholeCameraReprojectionError::extrinsic_parameter_count;

  inline auto initialize_intrinsics(const Eigen::Matrix3d& K) -> void
  {
    _intrinsics[0] = K(0, 0);
    _intrinsics[1] = K(0, 2);
    _intrinsics[2] = K(1, 2);
  }

  inline auto initialize_obs_3d(int w, int h, double square_size) -> void
  {
    _w = w;
    _h = h;
    _num_corners = w * h;

    for (auto y = 0; y < h; ++y)
    {
      for (auto x = 0; x < w; ++x)
      {
        _observations_3d.push_back(x * square_size);
        _observations_3d.push_back(y * square_size);
      }
    }
  }

  inline auto add(const sara::OpenCV::Chessboard& chessboard,
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
    for (auto y = 0; y < chessboard.height(); ++y)
    {
      for (auto x = 0; x < chessboard.width(); ++x)
      {
        const Eigen::Vector2f image_point = chessboard.image_point(x, y);
        _observations_2d.push_back(image_point.x());
        _observations_2d.push_back(image_point.y());
      }
    }
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
    for (auto n = 0; n < _num_images; ++n)
    {
      for (auto y = 0; y < _h; ++y)
      {
        for (auto x = 0; x < _w; ++x)
        {
          const auto corner_index = y * _w + x;
          const auto index = n * _num_corners + corner_index;

          const auto image_x =
              static_cast<double>(_observations_2d[2 * index + 0]);
          const auto image_y =
              static_cast<double>(_observations_2d[2 * index + 1]);
          const auto scene_x =
              static_cast<double>(_observations_3d[2 * corner_index + 0]);
          const auto scene_y =
              static_cast<double>(_observations_3d[2 * corner_index + 1]);

          auto cost_function = sara::PinholeCameraReprojectionError::create(  //
              image_x, image_y,                                               //
              scene_x, scene_y);

          problem.AddResidualBlock(  //
              cost_function,         //
              nullptr,               //
              mutable_intrinsics(),  //
              mutable_extrinsics(n)  //
          );
        }
      }
    }
  }

private:
  int _w = 0;
  int _h = 0;
  int _num_corners = 0;
  int _num_images = 0;

  std::vector<double> _observations_2d;
  std::vector<double> _observations_3d;
  std::array<double, intrinsic_parameter_count> _intrinsics;
  std::vector<double> _extrinsics;
};


GRAPHICS_MAIN()
{
  // #define SAMSUNG_GALAXY_J6
  // #define GOPRO4
#define IPHONE12
  // #define GOPRO7_WIDE
  // #define GOPRO7_SUPERVIEW

  const auto video_filepath =
#if defined(SAMSUNG_GALAXY_J6)
      "/home/david/Desktop/calibration/samsung-galaxy-j6/chessboard.mp4"
#elif defined(GOPRO4)
      "/home/david/Desktop/calibration/gopro-hero4/chessboard.mp4"
#elif defined(IPHONE12)
      "/home/david/Desktop/calibration/iphone12/chessboard.mov"
#elif defined(GOPRO7_WIDE)
      "/home/david/Desktop/calibration/gopro-hero-black-7/wide/GH010052.MP4"
#elif defined(GOPRO7_SUPERVIEW)
      "/home/david/Desktop/calibration/gopro-hero-black-7/superview/"
      "GH010053.MP4"
#  pragma error "INVALID!"
#endif
      ;

  auto video_stream = sara::VideoStream{video_filepath};
  auto frame = video_stream.frame();

#if defined(SAMSUNG_GALAXY_J6) || defined(GOPRO4) || defined(IPHONE12)
  static const auto pattern_size = Eigen::Vector2i{7, 5};
  static constexpr auto square_size = 3._cm;
#elif defined(GOPRO7_WIDE) || defined(GOPRO7_SUPERVIEW)
  static const auto pattern_size = Eigen::Vector2i{7, 9};
  static constexpr auto square_size = 2._cm;
#endif
  auto chessboards = std::vector<sara::OpenCV::Chessboard>{};

  // Initialize the calibration matrix.
  auto K = init_K(frame.width(), frame.height());

  // Initialize the calibration problem.
  auto calibration_problem = ChessboardCalibrationProblem{};
  calibration_problem.initialize_intrinsics(K);
  calibration_problem.initialize_obs_3d(pattern_size.x(), pattern_size.y(),
                                        square_size.value);


  static constexpr auto num_frames = 1000;
  auto selected_frames = std::vector<sara::Image<sara::Rgb8>>{};
  sara::create_window(frame.sizes());
  sara::set_antialiasing();
  for (auto i = 0; i < num_frames; ++i)
  {
    if (!video_stream.read())
      break;

    if (i % 5 != 0)
      continue;

    SARA_CHECK(i);

    sara::tic();
    auto chessboard = sara::OpenCV::Chessboard(pattern_size, square_size.value);
    const auto corners_found = chessboard.detect(frame);
    sara::toc("Chessboard corner detection");

    if (corners_found)
    {
      auto frame_copy = sara::Image<sara::Rgb8>{frame};
      draw_chessboard(frame_copy, chessboard);

      const Eigen::Matrix3d H = estimate_H(chessboard).normalized();
      auto Rs = std::vector<Eigen::Matrix3d>{};
      auto ts = std::vector<Eigen::Vector3d>{};
      auto ns = std::vector<Eigen::Vector3d>{};

      // This simple approach gives the best results.
      decompose_H_RQ_factorization(H, K, Rs, ts, ns);

      calibration_problem.add(chessboard, Rs[0], ts[0]);

      SARA_DEBUG << "\nRi =\n" << Rs[0] << std::endl;
      SARA_DEBUG << "\nti =\n" << ts[0] << std::endl;
      SARA_DEBUG << "\nni =\n" << ns[0] << std::endl;

      inspect(frame_copy, chessboard, K, Rs[0], ts[0]);
      sara::display(frame_copy);

      selected_frames.emplace_back(video_stream.frame());
      chessboards.emplace_back(std::move(chessboard));
    }
  }

  SARA_DEBUG << "Instantiating Ceres Problem..." << std::endl;
  auto problem = ceres::Problem{};
  calibration_problem.transform_into_ceres_problem(problem);

  SARA_DEBUG << "Solving Ceres Problem..." << std::endl;
  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  auto summary = ceres::Solver::Summary{};
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  const auto num_data_pts =
      chessboards.size() * chessboards.front().corner_count();
  const auto rms_init = std::sqrt(summary.initial_cost / num_data_pts);
  const auto rms_final = std::sqrt(summary.final_cost / num_data_pts);
  SARA_DEBUG << "RMS[INITIAL] = " << rms_init << std::endl;
  SARA_DEBUG << "RMS[FINAL  ] = " << rms_final << std::endl;

  const auto fx = calibration_problem.mutable_intrinsics()[0];
  const auto fy = calibration_problem.mutable_intrinsics()[1];
  const auto s = calibration_problem.mutable_intrinsics()[2];
  const auto u0 = calibration_problem.mutable_intrinsics()[3];
  const auto v0 = calibration_problem.mutable_intrinsics()[4];

  // clang-format off
  K(0, 0) = fx; K(0, 1) =  s; K(0, 2) = u0;
                K(1, 1) = fy; K(1, 2) = v0;
  // clang-format on
  SARA_DEBUG << "K =\n" << K << std::endl;

  for (auto i = 0u; i < chessboards.size(); ++i)
  {
    const auto R = calibration_problem.rotation(i).toRotationMatrix();
    const auto t = calibration_problem.translation(i);

    auto frame_copy = selected_frames[i];
    inspect(frame_copy, chessboards[i], K, R, t);
    sara::display(frame_copy);
    sara::get_key();
  }

  return 0;
}