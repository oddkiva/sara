// ========================================================================== //
// This file is part of Sara", a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Match/Match.hpp>


namespace DO { namespace Sara {

  //! @addtogroup Match
  //! @{
  class PairWiseDrawer
  {
  public:
    enum CatType
    {
      CatH,
      CatV
    };

    PairWiseDrawer(const ImageView<Rgb8>& I1, const ImageView<Rgb8>& I2)
      : image1(I1)
      , image2(I2)
    {
    }

    //! Set visualization parameters.
    inline void set_viz_params(float s1, float s2, CatType concatType)
    {
      _z1 = s1;
      _z2 = s2;
      _cat_type = concatType;
      _off2 = _cat_type == CatH ? Point2i(image1.width(), 0)
                                : Point2i(0, image1.height());
    }

    DO_SARA_EXPORT
    void display_images() const;

    DO_SARA_EXPORT
    void draw_point(int i, const Point2f& p, const Rgb8& c, int r = 2) const;

    DO_SARA_EXPORT
    void draw_line(int i, const Point2f& pa, const Point2f& pb, const Rgb8& c,
                   int penWidth = 1) const;

    DO_SARA_EXPORT
    void draw_arrow(int i, const Point2f& pa, const Point2f& pb, const Rgb8& c,
                    int penWidth = 1) const;

    DO_SARA_EXPORT
    void draw_triangle(int i, const Point2f& pa, const Point2f& pb,
                       const Point2f& pc, const Rgb8& c = Cyan8,
                       int r = 2) const;

    DO_SARA_EXPORT
    void draw_rect(int i, const Point2f& p1, const Point2f& p2, int r,
                   const Rgb8& c = Yellow8) const;

    template <typename LineIterator>
    void draw_lines(int i, LineIterator first, LineIterator last,
                    const Rgb8& c = Black8, int r = 2) const
    {
      assert(i == 0 || i == 1);
      for (LineIterator line = first; line != last; ++line)
        draw_line(i, line->first, line->second, c, r);
    }

    DO_SARA_EXPORT
    void draw_line_from_eqn(int i, const Vector3f& eqn, const Rgb8& c = Cyan8,
                            int r = 2) const;

    template <typename EqnIterator>
    inline void draw_lines_from_eqns(int i, EqnIterator first, EqnIterator last,
                                     const Rgb8& c = Cyan8, int r = 2) const
    {
      assert(i == 0 || i == 1);
      for (EqnIterator eqn = first; eqn != last; ++eqn)
        draw_line_from_eqn(i, *eqn, c, r);
    }

    template <typename VHIterator>
    inline void draw_vertices(int i, VHIterator first, VHIterator last,
                              int r = 2, const Rgb8& c = Yellow8) const
    {
      assert(i == 0 || i == 1);
      for (VHIterator vh = first; vh != last; ++vh)
        draw_point(i, Point2f((*vh)->point().x(), (*vh)->point().y()), c, r);
    }

    DO_SARA_EXPORT
    void draw_feature(int i, const OERegion& f, const Rgb8& c = Red8) const;

    DO_SARA_EXPORT
    void draw_match(const Match& m, const Rgb8& c = Magenta8,
                    bool drawLine = false) const;

    const ImageView<Rgb8>& image(int i) const
    {
      assert(i == 0 || i == 1);
      return (i == 0) ? image1 : image2;
    }

    Point2i off(int i) const
    {
      assert(i == 0 || i == 1);
      return (i == 0) ? Point2i::Zero() : _off2;
    }

    Point2f offset(int i) const
    {
      return off(i).cast<float>();
    }

    float scale(int i) const
    {
      assert(i == 0 || i == 1);
      return (i == 0) ? _z1 : _z2;
    }

  private:
    const ImageView<Rgb8>& image1;
    const ImageView<Rgb8>& image2;

    CatType _cat_type;
    Point2i _off2;
    float _z1 = 1.f;
    float _z2 = 1.f;
  };

  //! @}

}}  // namespace DO::Sara
