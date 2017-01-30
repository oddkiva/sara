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

  
  struct FancyColorPCA
  {
    FancyColorPCA(const Matrix3d& U, const Vector3d& S)
      : _U{U}
      , _S{S}
    {
    }

    void operator()(Image<Rgb64f>& in, const Vector3d& alpha) const
    {
      in.array() += _U * _S.asDiagonal() * alpha;
    }

    Matrix3d _U;
    Vector3d _S;
  };


} /* namespace Sara */
} /* namespace DO */
