#include <DO/Sara/MultiViewGeometry/Camera/PinholeCamera.hpp>


template <typename T>
using CameraMatrix = Eigen::Matrix<T, 3, 4>;


//! @brief Camera position and orientation w.r.t. some absolute reference frame.
template <typename T>
struct CameraPose
{
  //! @brief 3D rotation in quaternion representation since it is more
  //! economical in space.
  Eigen::Quaternion<T> q;
  //! @brief 3D translation.
  Eigen::Matrix<T, 3, 1> t;

  inline auto camera_matrix() const -> CameraMatrix<T>
  {
    return (CameraMatrix<T>{} << q.toRotationMatrix(), t).finished();
  }
};


//! @brief Estimate the height of the bounding box in meters.
/*!
 *  The calculation assumes a calibrated image:
 *  - the backprojected ray is computed from the camera model.
 *  - the backprojected ray is computed from the camera model.
 *
 */
template <typename T, typename CameraModel>
auto estimate_box_height(
    const Eigen::Matrix<T, 2, 1>& box_ground_coordinates,
    const Eigen::Matrix<T, 3, 1>& backprojected_ray_from_box_top_line,
    const CameraMatrix<T>& C) -> T
{
  const auto& x = box_ground_coordinates.x();
  const auto& y = box_ground_coordinates.y();

  const auto R1 = C.col(0);
  const auto R2 = C.col(1);
  const auto R3 = C.col(2);
  const auto t = C.col(3);

  const auto& r = backprojected_ray_from_box_top_line;

  const Eigen::Matrix<T, 3, 2> A =
      (Eigen::Matrix<T, 3, 2>{} << r, -R3).finished();
  const Eigen::Matrix<T, 3, 1> b = x * R1 + y * R2 + t;

  const Eigen::Matrix<T, 2, 1> solution =
      A.template colPivHouseholderQr().solve(b);

  const auto& scale = solution(0);
  const auto& height = solution(1);

  return height;
}


int main()
{
  return 0;
}
