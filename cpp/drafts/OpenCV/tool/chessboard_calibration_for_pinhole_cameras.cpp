#include <opencv2/opencv.hpp>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/Resectioning/HartleyZisserman.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/core/eigen.hpp>

#include <drafts/OpenCV/HomographyDecomposition.hpp>
#include <drafts/OpenCV/HomographyEstimation.hpp>


namespace sara = DO::Sara;

using sara::operator""_cm;


// This seems to work well...
inline auto init_K(int w, int h) -> Eigen::Matrix3d
{
  const auto d = static_cast<double>(std::max(w, h));
  const auto f = 0.5 * d;

  // clang-format off
  const auto K = (Eigen::Matrix3d{} <<
                  f, 0, w * 0.5,
                  0, f, h * 0.5,
                  0, 0,       1).finished();
  // clang-format on

  return K;
}

auto inspect(sara::ImageView<sara::Rgb8>& image,    //
             const sara::OpenCV::Chessboard& chessboard,  //
             const Eigen::Matrix3d& K,              //
             const Eigen::Matrix3d& R,              //
             const Eigen::Vector3d& t) -> void
{
  auto Hr = Eigen::Matrix3f{};
  Hr.col(0) = R.col(0).cast<float>();
  Hr.col(1) = R.col(1).cast<float>();
  Hr.col(2) = t.cast<float>();
  Hr = (K.cast<float>() * Hr).normalized();

  const auto a = chessboard(0, 0);
  const auto b = chessboard(0, 1);
  const auto c = chessboard(1, 0);

  const Eigen::Vector2f d = (K * (R * Eigen::Vector3d::UnitZ() + t))  //
                                .hnormalized()
                                .cast<float>();


  static const auto red = sara::Rgb8{167, 0, 0};
  sara::draw_arrow(image, a, b, red, 6);
  sara::draw_circle(image, a, 5.f, red, 6);
  sara::draw_circle(image, b, 5.f, red, 6);

  static const auto green = sara::Rgb8{89, 216, 26};
  sara::draw_arrow(image, a, c, green, 6);
  sara::draw_circle(image, a, 5.f, green, 6);
  sara::draw_circle(image, c, 5.f, green, 6);

  sara::draw_arrow(image, a, d, sara::Blue8, 6);
  sara::draw_circle(image, a, 5.f, sara::Blue8, 6);
  sara::draw_circle(image, d, 5.f, sara::Blue8, 6);

  for (auto i = 0; i < chessboard.height(); ++i)
  {
    for (auto j = 0; j < chessboard.width(); ++j)
    {
      const Eigen::Vector2f x = chessboard(i, j);

      const Eigen::Vector3f X = chessboard.point_3d(i, j).homogeneous();
      const Eigen::Vector2f x2 = (Hr * X).hnormalized();

      sara::draw_circle(image, x, 3.f, sara::Cyan8, 3);
      sara::draw_circle(image, x2, 3.f, sara::Magenta8, 3);
    }
  }
}

struct ReprojectionError
{
  static constexpr auto residual_dimension = 2;
  static constexpr auto intrinsic_parameter_count = 1;
  static constexpr auto extrinsic_parameter_count = 6;

  inline ReprojectionError(double imaged_x, double imaged_y,  //
                           double scene_x, double scene_y)
    : image_point{imaged_x, imaged_y}
    , scene_point{scene_x, scene_y}
  {
  }

  // We optimize:
  // - only one set of intrinsics
  // - as many set of extrinsics as there are images
  template <typename T>
  inline auto operator()(const T* const intrinsics, const T* const extrinsics,
                         T* residuals) const -> bool
  {
    // 1. Apply [R|t] = extrinsics[...]
    //
    // a) extrinsics[0, 1, 2] are the angle-axis rotation.
    auto scene_coords =
        std::array<T, 3>{T(scene_point.x()), T(scene_point.y()), T{}};
    auto camera_coords = std::array<T, 3>{};
    ceres::AngleAxisRotatePoint(extrinsics, scene_coords.data(),
                                camera_coords.data());

    // b) extrinsics[3, 4, 5] are the translation.
    camera_coords[0] += extrinsics[3];
    camera_coords[1] += extrinsics[4];
    camera_coords[2] += extrinsics[5];

    // 2. Calculate the normalized camera coordinates.
    const auto xp = camera_coords[0] / camera_coords[2];
    const auto yp = camera_coords[1] / camera_coords[2];

    // 3. Apply the calibration matrix.
    const auto& focal = intrinsics[0];
    // const auto& u0 = intrinsics[1];
    // const auto& v0 = intrinsics[2];
    const auto predicted_x = focal * xp + T(960);
    const auto predicted_y = focal * yp + T(540);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - static_cast<T>(image_point[0]);
    residuals[1] = predicted_y - static_cast<T>(image_point[1]);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static inline auto create(const double imaged_x, const double imaged_y,
                            const double scene_x, const double scene_y)
  {
    return new ceres::AutoDiffCostFunction<ReprojectionError,  //
                                           residual_dimension,
                                           intrinsic_parameter_count,
                                           extrinsic_parameter_count>(
        new ReprojectionError(imaged_x, imaged_y, scene_x, scene_y)  //
    );
  }

  Eigen::Vector2d image_point;
  Eigen::Vector2d scene_point;
};


class ChessboardCalibrationProblem
{
public:
  static constexpr auto intrinsic_parameter_count = 1;
  static constexpr auto extrinsic_parameter_count = 6;

  inline auto initialize_intrinsics(const Eigen::Matrix3d& K) -> void
  {
    _intrinsics[0] = K(0, 0);
    // _intrinsics[1] = K(0, 2);
    // _intrinsics[2] = K(1, 2);
    SARA_CHECK(_intrinsics[0]);
    // SARA_CHECK(_intrinsics[1]);
    // SARA_CHECK(_intrinsics[2]);
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
        SARA_DEBUG << "scene " << x << " " << y << std::endl;
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
    SARA_DEBUG << "angle = " << angle << std::endl;
    SARA_DEBUG << "axis = " << axis.transpose() << std::endl;
    SARA_DEBUG << "axis.norm = " << axis.norm() << std::endl;
    for (auto i = 0; i < 3; ++i)
      _extrinsics.push_back(angle * axis(i));

    // Initialize the translation.
    for (auto i = 0; i < 3; ++i)
      _extrinsics.push_back(t(i));

    // Store the image points.
    for (auto i = 0; i < chessboard.height(); ++i)
    {
      for (auto j = 0; j < chessboard.width(); ++j)
      {
        const Eigen::Vector2f image_point = chessboard(i, j);
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
    SARA_CHECK(_num_images);
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

          SARA_CHECK(n);
          SARA_CHECK(corner_index);
          SARA_DEBUG << "image: " << image_x << " " << image_y << std::endl;
          SARA_DEBUG << "scene: " << scene_x << " " << scene_y << std::endl;

          auto extrinsics = mutable_extrinsics(n);
          SARA_DEBUG << "angle-axis: "        //
                     << extrinsics[0] << " "  //
                     << extrinsics[1] << " "  //
                     << extrinsics[2] << std::endl;
          SARA_DEBUG << "translation: "           //
                     << extrinsics[3 + 0] << " "  //
                     << extrinsics[3 + 1] << " "  //
                     << extrinsics[3 + 2] << std::endl;
          sara::draw_circle(image_x, image_y, 3., sara::Blue8, 4);
          // sara::get_key();

          auto cost_function =
              ReprojectionError::create(image_x, image_y, scene_x, scene_y);
          problem.AddResidualBlock(cost_function, nullptr, mutable_intrinsics(),
                                   mutable_extrinsics(n));
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
// #define LUXVISION
#define GOPRO4
#define GOPRO7_WIDE
#define GOPRO7_SUPERVIEW
  // #define GOPRO4
  auto video_stream = sara::VideoStream{
#ifdef LUXVISION
      "/media/Linux Data/"
      "ha/safetytech/210330_FishEye/calibration_luxvision_cameras/"
      "checkboard_luxvision_1.MP4"
#elif defined(GOPRO4)
      "/home/david/Desktop/calibration/gopro-hero4/chessboard.mp4"
#elif defined(GOPRO7_WIDE)
      "/home/david/Desktop/calibration/gopro-hero-black-7/wide/GH010052.MP4"
#else
      "/home/david/Desktop/calibration/gopro-hero-black-7/superview/"
      "GH010053.MP4"
#endif
  };
  auto frame = video_stream.frame();

#if defined(LUXVISION)
  static const auto pattern_size = Eigen::Vector2i{7, 12};
  static constexpr auto square_size = 7._cm;
#elif defined(GOPRO4)
  static const auto pattern_size = Eigen::Vector2i{7, 5};
  static constexpr auto square_size = 3._cm;
#elif defined(GOPRO7_WIDE) || defined(GOPRO7_SUPERVIEW)
  static const auto pattern_size = Eigen::Vector2i{7, 9};
  static constexpr auto square_size = 2._cm;
#endif
  auto chessboards = std::vector<sara::OpenCV::Chessboard>{};

  auto K = init_K(frame.width(), frame.height());

  // Initialize the calibration problem.
  auto calibration_problem = ChessboardCalibrationProblem{};
  calibration_problem.initialize_intrinsics(K);
  calibration_problem.initialize_obs_3d(pattern_size.x(), pattern_size.y(), square_size.value);


  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  static constexpr auto num_frames = 90;
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
      draw_chessboard(frame, chessboard);

      const Eigen::Matrix3d H = estimate_H(chessboard).normalized();
      auto Rs = std::vector<Eigen::Matrix3d>{};
      auto ts = std::vector<Eigen::Vector3d>{};
      auto ns = std::vector<Eigen::Vector3d>{};

      // TODO: take the time to better understand the math underneath.
      // decompose_H_faugeras(H, K, Rs, ts, ns);

      // So far this simple approach gave the best results.
      decompose_H_RQ_factorization(H, K, Rs, ts, ns);

      calibration_problem.add(chessboard, Rs[0], ts[0]);

      auto frame_copy = sara::Image<sara::Rgb8>{frame};

#ifdef DEBUG_POSE_INITIALIZATION
      SARA_DEBUG << "\nRi =\n" << Rs[i] << std::endl;
      SARA_DEBUG << "\nti =\n" << ts[i] << std::endl;
      SARA_DEBUG << "\nni =\n" << ns[i] << std::endl;
      SARA_DEBUG << "H =\n" << H << "\n";
      SARA_DEBUG << "H_rec_i =\n" << Hr << "\n\n";
#endif

      inspect(frame_copy, chessboard, K, Rs[0], ts[0]);
      sara::display(frame_copy);
      // sara::get_key();

      chessboards.emplace_back(std::move(chessboard));
    }
  }

  auto problem = ceres::Problem{};
  calibration_problem.transform_into_ceres_problem(problem);

  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::DENSE_QR;  // ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  auto summary = ceres::Solver::Summary{};
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";


  const auto f = calibration_problem.mutable_intrinsics()[0];
  // const auto u0 = calibration_problem.mutable_intrinsics()[1];
  // const auto v0 = calibration_problem.mutable_intrinsics()[2];
  SARA_CHECK(f);
  // SARA_CHECK(u0);
  // SARA_CHECK(v0);

  K(0, 0) = f;
  K(1, 1) = f;

  auto i = 0;
  for (const auto& chessboard: chessboards)
  {
    const auto R = calibration_problem.rotation(i).toRotationMatrix();
    const auto t = calibration_problem.translation(i);
    SARA_DEBUG << "R =\n" << R << std::endl;
    SARA_DEBUG << "t =\n" << t << std::endl;

    auto frame_copy = sara::Image<sara::Rgb8>{frame};
    inspect(frame_copy, chessboard, K, R, t);
    sara::display(frame_copy);
    sara::get_key();

    ++i;
  }

  return 0;
}
