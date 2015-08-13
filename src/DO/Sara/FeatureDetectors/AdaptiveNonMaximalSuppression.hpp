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

#ifndef DO_SARA_FEATUREDETECTORS_ADAPTIVENONMAXIMALSUPPRESSION_HPP
#define DO_SARA_FEATUREDETECTORS_ADAPTIVENONMAXIMALSUPPRESSION_HPP

#include <vector>
#include <utility>

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup FeatureDetectors
    \defgroup ANMS Adaptive Non Maximal Suppression
    @{
  */

  /*!
    \brief Adaptive non maximal suppression algorithm (cf. [Multi-Image
    Matching using Multi-Scale Oriented Patches, Brown et al., CVPR 2005]).

    This is the naive implementation which is quadratic in the number of
    features. Because of its complexity, it does not scale well.

    Adaptive non maximal suppression is presented for the first time in:
      [Multi-Image Matching using Multi-Scale Oriented Patches, Brown et al.,
       CVPR 2005].
    It aims at discarding feature points so that the remaining features:
    - are as evenly spaced as possible;
    - are among the most distinctive set of points.

    Let \f$(f_i)_{1 \leq i \leq N}\f$ be the set of feature points to filter.
    In the sequel, we denote:
    - by \f$\mathbf{x}_i\f$ the position of feature \f$f_i\f$,
    - by \f$v_i\f$ the strength value of feature point \f$f_i\f$.

    For example, in the case of the Harris-Stephens corners, \f$v_i\f$ are
    local maximum values of the Harris-Stephens cornerness function.

    Specifically, the function computes for each feature \f$f_i\f$ the
    suppression radius defined as:
    \f[
      r_i = \min_{j \in I_i} \| \mathbf{x}_i - \mathbf{x}_j \|_2
    \f]
    where \f$I_i\f$ is the set of feature points \f$f_j\f$ with values
    \f$v_j\f$ stronger than value \f$v_i\f$ of feature \f$f_i\f$, i.e.,
    \f[
      I_i = \{ j \in \{1,\dots, n\} \mid v_i < c_\textrm{robust} v_j \}.
    \f]

    Note that \f$I_i\f$ can be empty, and in that then we set
    \f$r_i = \infty\f$.

    The intuition is the following. If \f$f_i\f$ is close to stronger
    features \f$f_j\f$ and has a value \f$v_i\f$ much lower than their values
    \f$v_j\f$, then \f$f_i\f$ should be suppressed. It is indeed reflected by
    the fact that its suppression radius \f$r_i\f$ is smaller than those of
    stronger features.

    The adaptive non maximal suprression sorts feature points by suppression
    radius and so that we can keep those with the highest supression radius.
   */
  DO_EXPORT
  std::vector<std::pair<size_t, float> >
  adaptive_non_maximal_suppression(const std::vector<OERegion>& features,
                                   float c_robust = 0.9f);

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDETECTORS_ADAPTIVENONMAXIMALSUPPRESSION_HPP */
