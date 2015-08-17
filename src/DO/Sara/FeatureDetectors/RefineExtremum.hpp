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

#ifndef DO_SARA_FEATUREDETECTORS_REFINEEXTREMA_HPP
#define DO_SARA_FEATUREDETECTORS_REFINEEXTREMA_HPP


namespace DO { namespace Sara {

  /*!
    \ingroup FeatureDetectors
    \defgroup ExtremumRefinement Extremum Localization Refinement
    @{
  */

  /*!
    \brief Test based on Harris-Stephens' idea on corner detection.
    Basically the Hessian matrix \f$\mathbf{H}\f$ is estimated by finite
    differentiation.

    \f$(x,y)\f$ is on an edge if the Hessian matrix \f$\mathbf{H}\f$ satisfies
    the following criterion
    \f$
      \frac{\mathrm{det}(\mathbf{H})}{\mathrm{tr}(\mathbf{H})} >
      \frac{(r+1)^2}{r}
    \f$,
    where \f$r\f$ is the ratio between the eigenvalues of \f$\mathbf{H}\f$
    and corresponds to the variable **edgeRatio**.
   */
  DO_EXPORT
  bool on_edge(const Image<float>& I, int x, int y, float edge_ratio = 10.f);

  /*!
    \brief Extremum position refinement in scale-space based on Newton's method.
    (cf. [Lowe, IJCV 2004] and [Brown and Lowe, BMVC 2002]).

    @param[in] I the input gaussian pyramid
    @param[in] x integral x-coordinate of the extrema
    @param[in] y integral y-coordinate of the extrema
    @param[in] s scale index of the extrema
    @param[in] o octave index of the extrema
    @param[in]
      img_padding_sz
      This variable indicates the minimum border size of the image. DoG
      extrema that ends being located the border are not refined anymore.
    @param[in]
      numIter
      This variable controls the number of iterations to refine the
      localization of DoG extrema in scale-space. The refinement process is
      based on the function **DO::refineExtremum()**.

    Let \f$D : \mathbf{R}^3 \mapsto \mathbf{R}\f$ be the difference of gaussian
    function and \f$(x,y,\sigma)\f$ be the approximate position
    of a local extremum of \f$D\f$.

    If \f$(x,y,\sigma)\f$ is the current guess of the local extremum, the refinement
    procedure seeks to minimize the following objective function iteratively:
    \f{eqnarray*}{
      \mathrm{minimize}_{\mathbf{h}}
          D(\mathbf{x})
        + D'(\mathbf{x})^T \mathbf{h}
        + 1/2 \mathbf{h}^T D''(\mathbf{x}) \mathbf{h}^T .
    \f}
    In practice the gradient vector \f$D'(\mathbf{x})\f$ and hessian matrix
    \f$D''(\mathbf{x})\f$ are approximated by finite difference and one must check
    that the hessian matrix \f$D''(\mathbf{x})\f$ is indeed **positive-definite**.

    Likewise, if \f$\mathbf{x}\f$ is a minimum, then one must check that
    \f$D''(\mathbf{x})\f$ is **negative-definite**.

    Otherwise, we cannot refine the position of the extremum.
   */
  DO_EXPORT
  bool refine_extremum(const ImagePyramid<float>& I,
                       int x, int y, int s, int o, int type,
                       Point3f& pos, float& val,
                       int border_size = 1, int num_iter = 5);
  /*!
    \brief This function refines the coordinates using the interpolation method
    in [Lowe, IJCV 2004] and [Brown and Lowe, BMVC 2002].

    It refines the spatial coordinates \f$(x,y)\f$. However, there is no scale
    refinement here.
   */
  DO_EXPORT
  bool refine_extremum(const Image<float>& I, int x, int y, int type,
                       Point2f& pos, float& val,
                       int border_size = 1, int num_iter = 5);
  /*!
    \brief Localizes all local extrema in scale-space at scale
    \f$\sigma = 2^{s/S+o}\f$.
    Note that the default parameters are suited for the DoG extrema.
   */
  DO_EXPORT
  std::vector<OERegion> local_scale_space_extrema(const ImagePyramid<float>& I,
                                                  int s, int o,
                                                  float extremum_thres = 0.03f,
                                                  float edge_ratio_thres = 10.f,
                                                  int img_padding_sz = 1,
                                                  int refine_iterations = 5);

  /*!
    Scale selection based on the normalized Laplacian of Gaussians
    for the simplified Harris-Laplace and Hessian-Laplace interest points.
   */
  DO_EXPORT
  bool select_laplace_scale(float& scale,
                            int x, int y, int s, int o,
                            const ImagePyramid<float>& gauss_pyramid,
                            int num_scales = 10);

  /*!
    \brief Localizes local maxima in space only and tries to assign a
    characteristic scale to each local maximum from the normalized Laplacian
    of Gaussians operator.

    This is mainly intended for Harris-Laplace and Hessian-Laplace interest
    points.
   */
  DO_EXPORT
  std::vector<OERegion> laplace_maxima(const ImagePyramid<float>& function,
                                       const ImagePyramid<float>& gaussian_pyramid,
                                       int s, int o,
                                       float extremum_thres = 1e-6f,
                                       int img_padding_sz = 1,
                                       int num_scales = 10,
                                       int refine_iterations = 5);

  //! @}

} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_FEATUREDETECTORS_REFINEEXTREMA_HPP */
