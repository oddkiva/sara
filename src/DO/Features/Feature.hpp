// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#ifndef DO_FEATURES_FEATURE_HPP
#define DO_FEATURES_FEATURE_HPP

namespace DO {

  //! Abstract 'Feature' class.
  class Feature {
  public:
    Feature() {}
    virtual ~Feature() {}
    virtual std::ostream& print(std::ostream& os) const = 0;
    virtual std::istream& read(std::istream& in) = 0;
  };
  inline std::ostream& operator<<(std::ostream& out, const Feature& obj)
  { return obj.print(out); }
  inline std::istream& operator>>(std::istream& in, Feature& obj)
  { return obj.read(in); }

  //! PointFeature is fairly a self-explanatory class.
  class PointFeature : public Feature
  {
  public:
    //! ID for each point feature type.
    enum Type { Harris, HarAff, HarLap, FAST, SUSAN,
                DoG, LoG, DoH, MSER, HesAff, HesLap };
    //! Constructors.
    PointFeature() : Feature() {}
    PointFeature(const Point2f& coords) : coords_(coords) {}
    //! Destructor.
    virtual ~PointFeature() {}
    //! Constant getters.
    float x() const { return coords_(0); }
    float y() const { return coords_(1); }
    const Point2f& coords() const { return coords_; }
    const Point2f& center() const { return coords(); }
    Type type() const { return type_; }
    //! Mutable getters.
    float& x() { return coords_(0); }
    float& y() { return coords_(1); }
    Point2f& coords() { return coords_; }
    Point2f& center() { return coords(); }
    Type& type() { return type_; }
    //! Equality operator.
    bool operator==(const PointFeature& f) const
    { return coords() == f.coords(); }
    //! I/O
    std::ostream& print(std::ostream& os) const
    { return os << x() << " " << y(); }
    std::istream& read(std::istream& in)
    { return in >> x() >> y(); }
  private:
    Point2f coords_;
    Type type_;
  };

  /*! The 'OERegion' class stands for 'Oriented Elliptic Region' and is 
    dedicated to store important geometric features such as:
    - DoG
    - Harris-Affine
    - Hessian-Affine features and so on...
   */
  class OERegion : public PointFeature
  {
  public:
    //! Default constructor
    OERegion() : PointFeature() {}
    //! Constructor for circular region.
    OERegion(const Point2f& coords, float scale)
      : PointFeature(coords), shape_matrix_(Matrix2f::Identity()*scale) {}
    //! Destructor.
    virtual ~OERegion() {}
    //! Constant/mutable shape matrix getters.
    //! The shape matrix is the matrix $M$ that describes the ellipse 
    //! $\varepsilon$, i.e.: 
    //! $$ \varepsilon = \{ x \in R^2 : (x-c)^T M (x-c) = 1 \} $$
    //! where $c$ is the center of the region.
    const Matrix2f& shapeMat() const { return shape_matrix_; }
    Matrix2f& shapeMat() { return shape_matrix_; }
    //! Constant/mutable orientation (in radian) getters.
    //! This completely determines the affine transformation that transforms the
    //! unit circle to the elliptic shape of the region.
    float orientation() const { return orientation_; }
    float& orientation() { return orientation_; }
    //! Returns the anisotropic radius at a given angle in radians.
    float radius(float radian = 0.f) const;
    //! Returns the anisotropic scale at a given angle in radians.
    float scale(float radian = 0.f) const { return radius(radian); }
    //! Get the affine transform $A$ that transforms the unit circle to that 
    //! oriented ellipse of the region.
    //! We compute $A$ from a QR decomposition and by observing
    //! $M = (A^{-1})^T A^{-1}$ where $M$ is the shape matrix.
    Matrix3f affinity() const;
    //! Equality operator.
    bool operator==(const OERegion& f) const
    {
      return (coords() == f.coords() &&
              shapeMat() == f.shapeMat() &&
              orientation() == f.orientation() &&
              type() == f.type());
    };
    //! Checks if the regions are equal regardless of the orientation.
    bool sameRegionAs(const OERegion& f) const
    {
      return (coords() == f.coords() &&
              shapeMat() == f.shapeMat() &&
              type() == f.type());
    }
    //! Lexicographical comparison by region.
    bool regionLexCompLess(const OERegion& f) const
    {
      if ( lexCompare(coords(), f.coords()) )
        return true;
      else if (coords() == f.coords() && lexCompare(shapeMat(), f.shapeMat()))
        return true;
      return false;
    }
    //! Lexicographical comparison by region and orientation.
    bool lexCompLess(const OERegion& f) const
    {
      if (regionLexCompLess(f))
        return true;
      else if (sameRegionAs(f) && orientation() < f.orientation())
        return true;
      return false;
    }
    //! I/O
    std::ostream& print(std::ostream& os) const
    { return PointFeature::print(os) << " " << shape_matrix_; }
    std::istream& read(std::istream& in)
    {
      return PointFeature::read(in) >> 
        shape_matrix_(0,0) >> shape_matrix_(0,1) >> 
        shape_matrix_(1,0) >> shape_matrix_(1,1);
    }
    //! Graphics.
    virtual void drawOnScreen(const Color3ub& c, float scale = 1.0f,
                              const Point2f& off = Point2f::Zero()) const;
  private:
    Matrix2f shape_matrix_;
    float orientation_;
  };

  class Corner : public OERegion
  {
  public:
    float corner_measure;
  };

  class Blob : public OERegion
  {
  public:
    enum ResponseType { Min, Max };

    Blob() : OERegion() {}
    Blob(const Point2f& coords, float scale) : OERegion(coords, scale) {}

    ResponseType& response() { return response_; }
    ResponseType response() const { return response_; }

  private:
    ResponseType response_;
    Type type_;
  };

}

#endif /* DO_FEATURES_FEATURE_HPP */