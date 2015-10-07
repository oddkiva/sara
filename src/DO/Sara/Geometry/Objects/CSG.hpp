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

#ifndef DO_SARA_GEOMETRY_OBJECTS_CSG_HPP
#define DO_SARA_GEOMETRY_OBJECTS_CSG_HPP

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara { namespace CSG {

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
    Obj _obj;

  public:
    explicit Singleton(const Obj& obj)
      : _obj(obj)
    {
    }

    bool contains(const Point2d& p) const
    {
      return _obj.contains(p);
    }
  };

  //! @brief CSG object difference.
  class Difference : public Object
  {
    const Object *_o1;
    const Object *_o2;

  public:
    inline Difference(const Object *o1, const Object *o2)
      : _o1(o1)
      , _o2(o2)
    {
    }

    bool contains(const Point2d& p) const
    {
      return _o1->contains(p) && !_o2->contains(p);
    }
  };

  //! @brief CSG object union.
  class Union : public Object
  {
    const Object *_o1;
    const Object *_o2;

  public:
    inline Union(const Object *o1, const Object *o2)
      : _o1(o1)
      , _o2(o2)
    {
    }

    bool contains(const Point2d& p) const
    {
      return _o1->contains(p) || _o2->contains(p);
    }
  };

  //! @brief CSG object intersection.
  class Intersection : public Object
  {
    const Object *_o1;
    const Object *_o2;

  public:
    inline Intersection(const Object *o1, const Object *o2)
      : _o1(o1)
      , _o2(o2)
    {
    }

    bool contains(const Point2d& p) const
    {
      return _o1->contains(p) && _o2->contains(p);
    }
  };

  //! @brief Computes the difference of two objects.
  inline Difference operator-(const Object& o1, const Object& o2)
  {
    return Difference(&o1, &o2);
  }

  //! @brief Computes the intersection of two objects.
  inline Intersection operator*(const Object& o1, const Object& o2)
  {
    return Intersection(&o1, &o2);
  }

  //! @brief Computes the union of two objects.
  inline Union operator+(const Object& o1, const Object& o2)
  {
    return Union(&o1, &o2);
  }

} /* namespace CSG */
} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GEOMETRY_OBJECTS_CSG_HPP */
