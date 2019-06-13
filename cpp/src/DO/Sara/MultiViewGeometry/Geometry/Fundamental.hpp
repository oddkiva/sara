#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  template <typename T = double>
  class FundamentalMatrix
  {
  public:
    using scalar_type = T;
    using matrix_type = Matrix<T, 3, 3>;
    using line_type = Matrix<T, 3, 1>;
    using point_type = Matrix<T, 3, 1>;

    FundamentalMatrix() = default;

    FundamentalMatrix(const matrix_type& m)
      : _m{m}
    {
    }

    auto matrix() const -> const matrix_type&
    {
      return _m;
    }

    auto matrix() -> matrix_type&
    {
      return _m;
    }

    operator const matrix_type&() const
    {
      return _m;
    }

    operator matrix_type&()
    {
      return _m;
    }

    auto epipoles() const -> std::pair<point_type, point_type>
    {
      auto svd = Eigen::JacobiSVD<matrix_type>{
          _m, Eigen::ComputeFullU | Eigen::ComputeFullV};

      const auto e1 = svd.matrixU().col(2).hnormalized();
      const auto e2 = svd.matrixV().col(2).hnormalized();

      return std::make_pair(e1, e2);
    }

    auto left_epipolar_line(const point_type& left) const -> line_type
    {
      return (_m * left).hnormalized();
    }

    auto right_epipolar_line(const point_type& right) const -> line_type
    {
      return (_m->transpose() * right).hnormalized();
    }

    auto is_rank_two() const -> bool
    {
      auto svd = JacobiSVD<matrix_type>{_m};
      return svd.singularValues()(2) == 0.;
    }

  protected:
    //! @brief Fundamental matrix container.
    matrix_type _m;
  };

} /* namespace Sara */
} /* namespace DO */
