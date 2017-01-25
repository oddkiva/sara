// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/Image/Image.hpp>

#include <DO/Sara/Features/Feature.hpp>

#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup FeatureDetectors
    @defgroup InterestPoint Interest Point Detection
    @{
   */

  //! Functor class to compute LoG extrema
  class DO_SARA_EXPORT ComputeLoGExtrema
  {
  public:
    /*!
      @brief Constructor
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
      const ImagePyramidParams& pyr_params = ImagePyramidParams(-1, 3+2),
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
      @brief Localizes LoG extrema for a given image.
     */
    std::vector<OERegion> operator()(const ImageView<float>& I,
                                     std::vector<Point2i> *scale_octave_pairs = 0);

    /*!
      @brief Returns the Gaussian pyramid used to localize scale-space extrema
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
      @brief Returns the pyramid of Laplacians of Gaussians used to localize
      scale-space extrema of image **I**.

      The pyramid of Laplacians of Gaussians is available after calling the
      function method **ComputeDoGExtrema::operator()(I, scale_octave_pairs)**,

      \return the pyramid of Laplacians of Gaussians used to localize
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& laplacians_of_gaussians() const
    {
      return _laplacians_of_gaussians;
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
    //! Gaussian-based image pyramids.
    ImagePyramid<float> _gaussians;
    ImagePyramid<float> _laplacians_of_gaussians;
    //! @}
  };

} /* namespace Sara */
} /* namespace DO */
