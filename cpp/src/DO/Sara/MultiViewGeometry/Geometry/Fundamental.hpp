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

    operator const matrix_type&() const
    {
      return _m;
    }

    operator matrix_type&()
    {
      return _m;
    }

    auto epipoles() const -> std::pair<matrix_type, matrix_type>
    {
      auto svd = Eigen::JacobiSVD<matrix_type>{
          _m, Eigen::ComputeFullU | Eigen::ComputeFullV};

      point_type e1 = svd.matrixU().col(2);
      e1 /= e1(2);

      point_type e2 = svd.matrixV().col(2);
      e2 /= e2(2);

      return std::make_pair(e1, e2);
    }

    auto left_epipolar_line(const point_type& x2) const -> line_type
    {
      line_type l = _m * x2;
      l /= l(2);
      return l;
    }

    auto right_epipolar_line(const point_type& x1) const -> line_type
    {
      line_type l = _m->transpose() * x1;
      l /= l(2);
      return l;
    }

    auto is_rank_2() const -> bool
    {
      auto svd = JacobiSVD<matrix_type>{_m};
      const auto predicate =
          svd.singularValues()(0) < std::numeric_limits<scalar_type>::epsilon();
      return predicate;
    }

  protected:
    //! @brief Fundamental matrix container.
    matrix_type _m;
  };


} /* namespace Sara */
} /* namespace DO */
