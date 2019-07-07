#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  class Homography
  {
  public:
    using matrix_type = Matrix3d;
    using line_type = Matrix3d;
    using point_type = Vector3d;

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
    matrix_type _m;
  };


} /* namespace Sara */
} /* namespace DO */
