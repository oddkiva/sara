#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>


namespace DO::Sara {

auto EssentialMatrix::extract_candidate_camera_motions() const
    -> std::array<EssentialMatrix::motion_type, 4>
{
  auto svd = JacobiSVD<matrix_type>{this->matrix(),
                                    Eigen::ComputeFullU | Eigen::ComputeFullV};

  //const auto U = svd.matrixU();
  //const auto W = svd.singularValues();
  //const auto V = svd.matrixU();
  //const auto t = svd.matrixU().col(2);

  //const Matrix3d R1 = U * W * V.transpose();
  //const Matrix3d R2 = U * W.transpose() * V.transpose();
  //const Vector3d t1 = t;
  //const Vector3d t2 = -t;

  //return {{R1, t1}, {R2, t1}, {R1, t2}, {R2, t2}};
  return {};
}

} /* namespace DO::Sara */
