// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Pixel.hpp>

#include <cstdint>
#include <stdexcept>


namespace DO { namespace Sara {

  /*!
   *  @addtogroup Features
   *  @{
   */

  //! @brief The oriented elliptic region class.
  /*!
   *  The 'OERegion' class stands for 'Oriented Elliptic Region' and is
   *  dedicated to store important geometric features such as:
   *  - DoG
   *  - Harris-Affine
   *  - Hessian-Affine features and so on...
   */
  class OERegion
  {
  public:
    //! @{
    //! @brief Feature type.
    enum class Type : std::uint8_t
    {
      Harris,
      HarAff,
      HarLap,
      FAST,
      SUSAN,
      DoG,
      LoG,
      DoH,
      MSER,
      HesAff,
      HesLap,
      Undefined
    };

    enum class ExtremumType : std::int8_t
    {
      Min = -1,
      Saddle = 0,
      Max = 1,
      Undefined = -2
    };
    //! @}

    //! @brief Default constructor
    OERegion() = default;

    //! @brief Constructor for circular region.
    OERegion(const Point2f& coords)
      : coords(coords)
    {
    }

    OERegion(const Point2f& coords, float scale)
      : coords{coords}
      , shape_matrix{Matrix2f::Identity() * (std::pow(scale, -2))}
    {
    }

    //! @{
    //! @brief Convenient getters.
    auto x() const -> float
    {
      return coords(0);
    }

    auto x() -> float&
    {
      return coords(0);
    }

    auto y() const -> float
    {
      return coords(1);
    }

    auto y() -> float&
    {
      return coords(1);
    }

    auto center() const -> const Point2f&
    {
      return coords;
    }

    auto center() -> Point2f&
    {
      return coords;
    }
    //! @}

    //! @brief Return the anisotropic radius at a given angle in radians.
    DO_SARA_EXPORT
    float radius(float radian = 0.f) const;

    //! @brief Return the anisotropic scale at a given angle in radians.
    float scale(float radian = 0.f) const
    {
      return radius(radian);
    }

    //! @brief Return the affine transform encoded by the shape matrix.
    /*!
     *  Get the affine transform $A$ that transforms the unit circle to that
     *  oriented ellipse of the region.
     *
     *  We compute $A$ from its QR decomposition and by observing that
     *  $M = (A^{-1})^T A^{-1}$ where $M$ is the shape matrix.
     */
    DO_SARA_EXPORT
    Matrix3f affinity() const;

    //! @brief Compare two regions.
    bool operator==(const OERegion& other) const
    {
      return (coords == other.coords &&              //
              shape_matrix == other.shape_matrix &&  //
              orientation == other.orientation &&    //
              type == other.type);
    };

    DO_SARA_EXPORT
    friend std::ostream& operator<<(std::ostream&, const OERegion&);

    DO_SARA_EXPORT
    friend std::istream& operator>>(std::istream&, OERegion&);

    //! @brief Center of the feature.
    Point2f coords{Point2f::Zero()};

    //! @brief Shape matrix encoding the ellipticity of the region.
    /*!
     *  The shape matrix is the matrix $M$ that describes the ellipse
     *  $\varepsilon$, i.e.:
     *  $$ \varepsilon = \{ x \in R^2 : (x-c)^T M (x-c) = 1 \} $$
     *  where $c$ is the center of the region.
     */
    Matrix2f shape_matrix{Matrix2f::Zero()};

    //! @brief Orientation of the region **after** shape normalization.
    /*!
     *  This completely determines the affine transformation that transforms the
     *  unit circle to the elliptic shape of the region.
     */
    float orientation{0};

    //! @{
    //! @brief Characterization of the feature type.
    float extremum_value{0};
    Type type{Type::Undefined};
    ExtremumType extremum_type{ExtremumType::Undefined};
    //! @}
  };

  //! @}

}}  // namespace DO::Sara
