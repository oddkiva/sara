// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file


#ifndef DO_SARA_FEATURES_FEATURE_HPP
#define DO_SARA_FEATURES_FEATURE_HPP

#include <cstdint>
#include <stdexcept>

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/Pixel.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup Features
    @{
  */

  //! \brief Abstract 'VisualFeature' class.
  class DO_EXPORT VisualFeature
  {
  public:
    VisualFeature() = default;

    virtual ~VisualFeature()
    {
    }

    virtual std::ostream& print(std::ostream& os) const = 0;

    virtual std::istream& read(std::istream& in) = 0;

    friend std::ostream& operator<<(std::ostream& out, const VisualFeature& f)
    {
      return f.print(out);
    }

    friend std::istream& operator>>(std::istream& in, VisualFeature& f)
    {
      return f.read(in);
    }
  };


  //! \brief PointFeature for interest points
  class DO_EXPORT InterestPoint : public VisualFeature
  {
  public:
    //! @{
    //! \brief Feature type.
    enum class Type : std::int8_t
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
      HesLap
    };

    enum class ExtremumType : std::int8_t
    {
      Min = -1,
      Saddle = 0,
      Max = 1
    };
    //! @}

    //! @{
    //! \brief Constructor.
    InterestPoint() = default;

    InterestPoint(const Point2f& coords)
      : _coords(coords)
    {
    }
    //! @}

    //! \brief Destructor.
    virtual ~InterestPoint()
    {
    }

    //! @{
    //! \brief Constant getters.
    float x() const
    {
      return _coords(0);
    }

    float y() const
    {
      return _coords(1);
    }

    const Point2f& coords() const
    {
      return _coords;
    }

    const Point2f& center() const
    {
      return coords();
    }

    Type type() const
    {
      return _type;
    }

    ExtremumType extremum_type() const
    {
      return _extremum_type;
    }

    float extremum_value() const
    {
      return _extremum_value;
    }
    //! @}

    //! \brief Mutable getters.
    float& x()
    {
      return _coords(0);
    }

    float& y()
    {
      return _coords(1);
    }

    Point2f& coords()
    {
      return _coords;
    }

    Point2f& center()
    {
      return coords();
    }

    Type& type()
    {
      return _type;
    }

    ExtremumType& extremum_type()
    {
      return _extremum_type;
    }

    float& extremum_value()
    {
      return _extremum_value;
    }
    //! @}

    //! \brief Equality operator.
    bool operator==(const InterestPoint& f) const
    {
      return coords() == f.coords();
    }

    //! \brief Draw feature.
    void draw(const Color3ub& c, float scale = 1.f,
              const Point2f& offset = Point2f::Zero()) const;

    //! @{
    //! I/O.
    std::ostream& print(std::ostream& os) const;

    std::istream& read(std::istream& in);
    //! @}

  private:
    Point2f _coords;
    Type _type;
    ExtremumType _extremum_type;
    float _extremum_value;
  };


  /*!
    The 'OERegion' class stands for 'Oriented Elliptic Region' and is
    dedicated to store important geometric features such as:
    - DoG
    - Harris-Affine
    - Hessian-Affine features and so on...
   */
  class DO_EXPORT OERegion : public InterestPoint
  {
  public:
    //! \brief Default constructor
    OERegion() = default;

    //! \brief Constructor for circular region.
    OERegion(const Point2f& coords, float scale)
      : InterestPoint{ coords }
      , _shape_matrix{ Matrix2f::Identity()*(pow(scale,-2)) }
    {
    }

    //! \brief Destructor.
    virtual ~OERegion()
    {
    }

    //! @{
    //! \brief Shape matrix accessor.
    //! The shape matrix is the matrix $M$ that describes the ellipse
    //! $\varepsilon$, i.e.:
    //! $$ \varepsilon = \{ x \in R^2 : (x-c)^T M (x-c) = 1 \} $$
    //! where $c$ is the center of the region.
    const Matrix2f& shape_matrix() const
    {
      return _shape_matrix;
    }

    Matrix2f& shape_matrix()
    {
      return _shape_matrix;
    }
    //! @}

    //! @{
    //! \brief Orientation (in radian) accessor.
    //! This completely determines the affine transformation that transforms the
    //! unit circle to the elliptic shape of the region.
    float orientation() const
    {
      return _orientation;
    }

    float& orientation()
    {
      return _orientation;
    }
    //! @}

    //! \brief Return the anisotropic radius at a given angle in radians.
    float radius(float radian = 0.f) const;

    //! \brief Return the anisotropic scale at a given angle in radians.
    float scale(float radian = 0.f) const
    {
      return radius(radian);
    }

    //! Get the affine transform $A$ that transforms the unit circle to that
    //! oriented ellipse of the region.
    //! We compute $A$ from a QR decomposition and by observing
    //! $M = (A^{-1})^T A^{-1}$ where $M$ is the shape matrix.
    Matrix3f affinity() const;

    //! \brief Compare two regions.
    bool operator==(const OERegion& other) const
    {
      return (coords() == other.coords() &&
              shape_matrix() == other.shape_matrix() &&
              orientation() == other.orientation() &&
              type() == other.type());
    };

    //! \brief Draw the region.
    void draw(const Color3ub& c, float scale = 1.f,
              const Point2f& offset = Point2f::Zero()) const;

    //! @{
    //! I/O
    std::ostream& print(std::ostream& os) const;

    std::istream& read(std::istream& in);
    //! @}

  private:
    //! \brief Shape matrix encoding the ellipticity of the region.
    Matrix2f _shape_matrix;
    //! \brief Orientation of the region **after** shape normalization.
    float _orientation;
  };


} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATURES_FEATURE_HPP */
