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

#ifndef DO_SARA_FEATUREDETECTORS_LOG_HPP
#define DO_SARA_FEATUREDETECTORS_LOG_HPP


namespace DO { namespace Sara {

  /*!
    \ingroup FeatureDetectors
    \defgroup InterestPoint Interest Point Detection
    @{
  */

  //! Functor class to compute LoG extrema
  class ComputeLoGExtrema
  {
  public:
    /*!
      \brief Constructor
      @param[in]
        extremum_thres
        the response threshold which the LoG extremum absolute value
        \f$
          \left|
            \sigma^2 \Delta I(\mathbf{x}, \sigma)
          \right|
        \f$
        must exceed.
        Note that \f$ \sigma(s',o') = 2^{s'/S+o'}\f$  where \f$S\f$ is the
        number of scales per octave.
      @param[in]
        edgeRatioThres
        the Hessian matrix \f$\mathbf{H}\f$ at the local scale-space extremum
        must satisfy
        \f[
          \frac{\mathrm{det}(\mathbf{H})}{\mathrm{tr}(\mathbf{H}} >
          \frac{(r+1)^2}{r}
        \f]
        where \f$r\f$ corresponds to the variable **edgeRatioThres**.
        In terms of implementation, we use the function **DO::onEdge()**.
        We use the \f$r=10\f$ as stated in [Lowe, IJCV 2004].
      @param[in]
        img_padding_sz
        This variable indicates the minimum border size of the image. DoG
        extrema located at the size-specified border are discarded.
      @param[in]
        extremumRefinementIter
        This variable controls the number of iterations to refine the
        localization of DoG extrema in scale-space. The refinement process is
        based on the function **DO::refineExtremum()**.
     */
    ComputeLoGExtrema(
      const ImagePyramidParams& pyrParams = ImagePyramidParams(-1, 3+2),
      float extremum_thres = 0.01f,
      float edgeRatioThres = 10.f,
      int img_padding_sz = 1,
      int extremumRefinementIter = 5)
      : params_(pyrParams)
      , extremum_thres_(extremum_thres)
      , edge_ratio_thres_(edgeRatioThres)
      , img_padding_sz_(img_padding_sz)
      , extremum_refinement_iter_(extremumRefinementIter)
    {}
    /*!
      \brief Localizes LoG extrema for a given image.
     */
    std::vector<OERegion> operator()(const Image<float>& I,
                                     std::vector<Point2i> *scaleOctavePairs = 0);
    /*!
      \brief Returns the Gaussian pyramid used to localize scale-space extrema
      of image **I**.

      The Gaussian pyramid is available after calling the function method
      **ComputeDoGExtrema::operator()(I, scaleOctavePairs)** for the given
      image **I**.

      \return the Gaussian pyramid used to localize scale-space extrema
      of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    { return gaussians_; }
    /*!
      \brief Returns the pyramid of Laplacians of Gaussians used to localize
      scale-space extrema of image **I**.

      The pyramid of Laplacians of Gaussians is available after calling the
      function method **ComputeDoGExtrema::operator()(I, scaleOctavePairs)**,

      \return the pyramid of Laplacians of Gaussians used to localize
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& laplaciansOfGaussians() const
    { return laplacians_of_gaussians_; }
  private: /* data members. */
    // Parameters
    ImagePyramidParams params_;
    float extremum_thres_;
    float edge_ratio_thres_;
    int img_padding_sz_;
    int extremum_refinement_iter_;
    // Difference of Gaussians.
    ImagePyramid<float> gaussians_;
    ImagePyramid<float> laplacians_of_gaussians_;
  };

} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_FEATUREDETECTORS_LOG_HPP */
