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

#pragma once

#include <DO/Sara/Core/EigenExtension.hpp>

namespace DO { namespace CSG {

  class Object
  {
  public:
    virtual ~Object()
    {
    }

    virtual bool contains(const Point2d& p) const = 0;
  };

  template <typename Obj>
  class Singleton : public Object
  {
    Obj obj_;
  public:
    explicit Singleton(const Obj& obj) : obj_(obj) {}
    bool contains(const Point2d& p) const
    { return inside(p, obj_); }
  };

  //! CSG object difference
  class Difference : public Object
  {
    const Object *o1_;
    const Object *o2_;
  public:
    inline Difference(const Object *o1, const Object *o2)
      : o1_(o1), o2_(o2) {}
    bool contains(const Point2d& p) const
    { return o1_->contains(p) && !o2_->contains(p); }
  };

  //! CSG object union
  class Union : public Object
  {
    const Object *o1_;
    const Object *o2_;
  public:
    inline Union(const Object *o1, const Object *o2)
      : o1_(o1), o2_(o2) {}
    bool contains(const Point2d& p) const
    { return o1_->contains(p) || o2_->contains(p); }
  };

  //! CSG object intersection
  class Intersection : public Object
  {
    const Object *o1_;
    const Object *o2_;
  public:
    inline Intersection(const Object *o1, const Object *o2)
      : o1_(o1), o2_(o2) {}
    bool contains(const Point2d& p) const
    { return o1_->contains(p) && o2_->contains(p); }
  };

  //! Difference of two objects
  inline Difference operator-(const Object& o1, const Object& o2)
  { return Difference(&o1, &o2); }

  //! Intersection of two objects
  inline Intersection operator*(const Object& o1, const Object& o2)
  { return Intersection(&o1, &o2); }

  //! Union of two objects
  inline Union operator+(const Object& o1, const Object& o2)
  { return Union(&o1, &o2); }

} /* namespace CSG */
} /* namespace DO */
