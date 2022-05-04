#include "OpenCVInterop.hpp"


namespace sara = DO::Sara;


auto estimate_H(const sara::OpenCV::Chessboard& chessboard) -> Eigen::Matrix3d
{
  const auto w = chessboard.width();
  const auto h = chessboard.height();
  const auto N = chessboard.corner_count();

  auto A = Eigen::MatrixXd{N * 2, 9};

  auto x = Eigen::MatrixXd{3, N};
  auto y = Eigen::MatrixXd{3, N};

  // Collect the 2D pixel coordinates.
  for (auto i = 0; i < h; ++i)
    for (auto j = 0; j < w; ++j)
      x.col(i * w + j) = chessboard(i, j).homogeneous().cast<double>();

  // Keep it simple by just divide by 1000. Lazy but it works.
  //
  // clang-format off
  const auto T = (Eigen::Matrix3d{} <<
                  1e-3,    0, 0,
                     0, 1e-3, 0,
                     0,    0, 1).finished();
  // clang-format on
  const Eigen::Matrix3d invT = T.inverse();

  // Rescale the coordinates.
  x = T * x;

  // Collect the 3D coordinates on the chessboard plane.
  for (auto i = 0; i < h; ++i)
    for (auto j = 0; j < w; ++j)
      y.col(i * w + j) = chessboard.point_3d(i, j).homogeneous().cast<double>();

  // Form the data matrix used to determine H.
  for (auto i = 0; i < N; ++i)
  {
    // The image point
    const auto xi = x.col(i);
    const auto ui = xi(0);
    const auto vi = xi(1);

    // The 3D coordinate on the chessboard plane.
    static const auto zero = Eigen::RowVector3d::Zero();
    const auto yiT = y.col(i).transpose();

    A.row(2 * i + 0) << -yiT, zero, ui * yiT;
    A.row(2 * i + 1) << zero, -yiT, vi * yiT;
  }

  // SVD.
  const auto svd = A.jacobiSvd(Eigen::ComputeFullV);
  const auto V = svd.matrixV();
  const auto H_flat = V.col(8);
  SARA_DEBUG << "V =\n" << V << std::endl;
  SARA_DEBUG << "H_flat =\n" << H_flat.transpose() << std::endl;

  // clang-format off
  auto H = Eigen::Matrix3d{};
  H << H_flat.head(3).transpose(),
       H_flat.segment(3, 3).transpose(),
       H_flat.tail(3).transpose();
  // clang-format on
  SARA_DEBUG << "H =\n" << H << std::endl;

  H = invT * H;
  SARA_DEBUG << "H =\n" << H << std::endl;

  return H;
}
