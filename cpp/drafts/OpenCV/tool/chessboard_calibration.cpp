#include <opencv2/opencv.hpp>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/Camera/OmnidirectionalCamera.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "../OpenCVInterop.hpp"


namespace sara = DO::Sara;
using sara::operator""_cm;


template <typename T>
struct OmnidirectionalCameraParamView
{
  inline OmnidirectionalCameraParamView() = default;

  inline OmnidirectionalCameraParamView(const T* const params)
    : params{params}
  {
  }

  // 5 -> K
  inline auto fx() -> T&
  {
    return params[0];
  }

  inline auto fy() -> const T&
  {
    return params[1];
  }

  inline auto shear() -> const T&
  {
    return params[2];
  }

  inline auto ppx() -> const T&
  {
    return params[3];
  }

  inline auto ppy() -> const T&
  {
    return params[3];
  }

  // 1 -> xi
  inline auto xi() -> const T&
  {
    return params[4];
  }

  // 3 -> radial
  inline auto radial_distortion_coeffs() -> const T*
  {
    return params + 4;
  }

  // 2 -> tangential
  inline auto tangential_distortion_coeffs() -> const T*
  {
    return params + 7;
  }

  // 3 -> translation
  inline auto translation() -> const T*
  {
    return params + 9;
  }

  // 3 -> angle axis
  inline auto angle_axis() -> const T*
  {
    return params + 12;
  }

  const T* params = nullptr;
};


struct ReprojectionError
{
  ReprojectionError(double observed_x, double observed_y)
    : observed_x{observed_x}
    , observed_y{observed_y}
  {
  }

  template <typename T>
  bool operator()(const T* const camera,  // (1) camera parameters to optimize.
                  const T* const point,   // (2) 3D points to optimize
                  T* residuals) const
  {
    T p[3];

    auto camera_view = OmnidirectionalCameraParamView<T>{camera};

    // Rotate the point.
    ceres::AngleAxisRotatePoint(camera_view.angle_axis(), point, p);
    // Translate the point.
    p[0] += camera_view.translation()[0];
    p[1] += camera_view.translation()[1];
    p[2] += camera_view.translation()[2];

    // Lifting.
    const auto p_norm = sqrt(sara::square(p[0]) +  //
                             sara::square(p[1]) +  //
                             sara::square(p[2]));
    p[0] /= p_norm;
    p[1] /= p_norm;
    p[2] /= p_norm;


    // residuals[0] = predicted_x - T(observed_x);
    // residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y)
  {
    constexpr auto NumParams = 15 /* camera parameters */ + 3 /* points */;
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, NumParams, 3>{
        new ReprojectionError{observed_x, observed_y}};
  }

  double observed_x;
  double observed_y;
};


template <typename T>
struct Optimizer
{
  inline explicit Optimizer(sara::OmnidirectionalCamera<T>& camera)
    : camera{camera}
  {
  }

  sara::OmnidirectionalCamera<T>& camera;
  std::vector<sara::OpenCV::Chessboard> chessboards;
};


GRAPHICS_MAIN()
{
  auto video_stream = sara::VideoStream{
      "/home/david/Desktop/calibration/gopro-hero-black-7/"  //
      "superview/GH010053.MP4"                               //
  };
  auto frame = video_stream.frame();

  static const auto pattern_size = Eigen::Vector2i{9, 7};
  static constexpr auto square_size = 2._cm;
  auto chessboard = sara::OpenCV::Chessboard(pattern_size, square_size.value);

  auto camera = sara::OmnidirectionalCamera<double>{};
  auto optimizer = Optimizer{camera};

  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  while (video_stream.read())
  {
    sara::tic();
    const auto corners_found = chessboard.detect(frame);
    sara::toc("Chessboard corner detection");

    if (corners_found)
      draw_chessboard(frame, chessboard);

    sara::display(frame);
    sara::get_key();
  }

  return 0;
}
