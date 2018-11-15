#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  template <typename T = double>
  class Homography
  {
  public:
    using scalar_type = T;
    using matrix_type = Matrix<T, 3, 3>;
    using line_type = Matrix<T, 3, 1>;
    using point_type = Matrix<T, 3, 1>;

    Homography() = default;

    Homography(const matrix_type& m)
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

  protected:
    //! @brief Fundamental matrix container.
    matrix_type _m;
  };


} /* namespace Sara */
} /* namespace DO */
