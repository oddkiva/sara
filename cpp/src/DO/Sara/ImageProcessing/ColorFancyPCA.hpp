// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageIO.hpp>


namespace DO { namespace Sara {


  struct ColorFancyPCA
  {
    ColorFancyPCA(const Matrix3f& U, const Vector3f& S)
      : _U{U}
      , _S{S}
    {
    }

    void operator()(const Image<Rgb32f>& in, Image<Rgb32f>& out, const Vector3f& alpha) const
    {
      out = in;
      for (int i = 0; i < out.size(); ++i)
      {
        //Vector3f p = _U * _S.asDiagonal() * alpha;
        //out.data()[i] += p;
      }
    }

    Matrix3f _U;
    Vector3f _S;
  };


} /* namespace Sara */
} /* namespace DO */
