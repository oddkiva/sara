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

//! @file

#ifndef DO_FEATUREDETECTORS_HESSIAN_HPP
#define DO_FEATUREDETECTORS_HESSIAN_HPP

namespace DO {

  /*!
    \ingroup InterestPoint
    @{
  */

  //! Computes a pyramid of determinant of Hessian from the Gaussian pyramid.
  template <typename T>
  ImagePyramid<T> DoHPyramid(const ImagePyramid<T>& gaussians)
  {
    ImagePyramid<T> D;
    D.reset(gaussians.numOctaves(),
            gaussians.numScalesPerOctave(),
            gaussians.initScale(), 
            gaussians.scaleGeomFactor());

    for (int o = 0; o < D.numOctaves(); ++o)
    {
      D.octaveScalingFactor(o) = gaussians.octaveScalingFactor(o);
      for (int s = 0; s < D.numScalesPerOctave(); ++s)
        D(s,o) = gaussians(s,o).
          template compute<Hessian>().
          template compute<Determinant>();
    }
    return D;
  }

  class ComputeDoHExtrema;
  class ComputeHessianLaplaceExtrema;

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDETECTORS_HESSIAN_HPP */
