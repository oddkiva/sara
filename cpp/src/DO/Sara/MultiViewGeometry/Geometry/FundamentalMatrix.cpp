#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>


namespace DO::Sara {

auto FundamentalMatrix::extract_epipoles() const
    -> std::tuple<point_type, point_type>
{
  auto svd = Eigen::JacobiSVD<matrix_type>{_m, Eigen::ComputeFullU |
                                                   Eigen::ComputeFullV};

  const auto right = svd.matrixU().col(2).hnormalized().homogeneous();
  const auto left = svd.matrixV().col(2).hnormalized().homogeneous();

  return {left, right};
}

auto FundamentalMatrix::right_epipolar_line(const point_type& left) const
    -> line_type
{
  return (_m * left).hnormalized().homogeneous();
}

auto FundamentalMatrix::left_epipolar_line(const point_type& right) const
    -> line_type
{
  return (_m.transpose() * right).hnormalized().homogeneous();
}


auto FundamentalMatrix::rank_two_predicate() const -> bool
{
  auto svd = Eigen::JacobiSVD<matrix_type>{_m};
  return std::abs(svd.singularValues()(2)) <
         std::numeric_limits<double>::epsilon();
}


std::ostream& operator<<(std::ostream& os, const FundamentalMatrix& f)
{
  return os << f.matrix() << std::endl;
}

} /* namespace DO::Sara */
