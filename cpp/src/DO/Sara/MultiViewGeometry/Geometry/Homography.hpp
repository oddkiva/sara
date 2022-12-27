#pragma once

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/EigenExtension.hpp>

#include <iostream>


namespace DO { namespace Sara {

  //! @addtogroup MultiViewGeometry
  //! @{

  class Homography
  {
  public:
    using matrix_type = Matrix3d;
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

    const auto& matrix() const
    {
      return _m;
    }

    auto& matrix()
    {
      return _m;
    }

  protected:
    matrix_type _m;
  };


  DO_SARA_EXPORT
  std::ostream& operator<<(std::ostream&, const Homography&);

  //! @}

} /* namespace Sara */
} /* namespace DO */
