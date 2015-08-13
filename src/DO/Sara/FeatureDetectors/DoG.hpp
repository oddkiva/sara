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

#ifndef DO_SARA_FEATUREDETECTORS_DOG_HPP
#define DO_SARA_FEATUREDETECTORS_DOG_HPP

#include <DO/Sara/Defines.hpp>


namespace DO { namespace Sara {

  /*!
    \ingroup FeatureDetectors
    \defgroup InterestPoint Interest Point Detection
    @{
  */

  //! Functor class to compute DoG extrema
  class DO_EXPORT ComputeDoGExtrema
  {
  public:
    /*!
      \brief Constructor
      @param[in]
        extremum_thres
        the response threshold which the DoG extremum absolute value
        \f$
          \left|
            \left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I
          \right| (\mathbf{x})
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
    ComputeDoGExtrema(
      const ImagePyramidParams& pyr_params = ImagePyramidParams(),
      float extremum_thres = 0.01f,
      float edge_ratio_thres = 10.f,
      int img_padding_sz = 1,
      int extremum_refinement_iter = 5)
      : _params(pyr_params)
      , _extremum_thres(extremum_thres)
      , _edge_ratio_thres(edge_ratio_thres)
      , _img_padding_sz(img_padding_sz)
      , _extremum_refinement_iter(extremum_refinement_iter)
    {
    }

    /*!
      \brief Localizes DoG extrema for a given image.

      This function does the following:
      1. Constructs a gaussian pyramid \f$\nabla g_{\sigma(s,o)} * I\f$ from
      the image \f$I\f$, where \f$(s,o)\f$ are integers. Here:
      \f$\sigma(s,o) = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per
      octaves.

      2. Localize extrema in each difference of Gaussians
      \f$\left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I \f$
      where \f$(s,o)\f$ are scale and octave indices.

      \param[in, out] scale_octave_pairs a pointer to vector of scale and octave
      index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the difference
      of Gaussians
      \f$\left( g_{\sigma(s_i+1,o_i)} - g_{\sigma(s_i,o_i)} \right) * I\f$
      where the extremum \f$(x_i,y_i,\sigma_i)\f$ is detected.

      \return set of DoG extrema in **std::vector<OERegion>** in each
      difference of Gaussians
      \f$\left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I \f$.
     */
    std::vector<OERegion> operator()(const Image<float>& I,
                                     std::vector<Point2i> *scale_octave_pairs = 0);

    /*!
      \brief Returns the Gaussian pyramid used to localize scale-space extrema
      of image **I**.

      The Gaussian pyramid is available after calling the function method
      **ComputeDoGExtrema::operator()(I, scale_octave_pairs)** for the given
      image **I**.

      \return the Gaussian pyramid used to localize scale-space extrema
      of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    {
      return _gaussians;
    }

    /*!
      \brief Returns the pyramid of difference of Gaussians used to localize
      scale-space extrema of image **I**.

      The pyramid of difference of Gaussians is available after calling the
      function method **ComputeDoGExtrema::operator()(I, scale_octave_pairs)**,

      \return the pyramid of difference of Gaussians used to localize
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& diff_of_gaussians() const
    {
      return _diff_of_gaussians;
    }

  private: /* data members. */
    //! @{
    //! Parameters
    ImagePyramidParams _params;
    float _extremum_thres;
    float _edge_ratio_thres;
    int _img_padding_sz;
    int _extremum_refinement_iter;
    //! @}

    //! @{
    //! Difference of Gaussians.
    ImagePyramid<float> _gaussians;
    ImagePyramid<float> _diff_of_gaussians;
    //! @}
  };


} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDETECTORS_DOG_HPP */