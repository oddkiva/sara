// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  //! @ingroup MultiViewGeometry
  //! @defgroup FundamentalMatrix Fundamental Matrix
  //! @{

  //! @brief Fundamental matrix API
  //! @{
  class DO_SARA_EXPORT FundamentalMatrix
  {
  public:
    using matrix_type = Eigen::Matrix3d;
    using vector_type = Eigen::Vector3d;
    using point_type = vector_type;
    using line_type = vector_type;


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

    auto extract_epipoles() const -> std::tuple<point_type, point_type>;

    auto right_epipolar_line(const point_type& left) const -> line_type;

    auto left_epipolar_line(const point_type& right) const -> line_type;

    auto rank_two_predicate() const -> bool;

  protected:
    //! @brief Fundamental matrix container.
    matrix_type _m;
  };


  DO_SARA_EXPORT
  std::ostream& operator<<(std::ostream&, const FundamentalMatrix&);
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */
