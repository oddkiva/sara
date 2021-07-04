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

#include <DO/Sara/ImageProcessing/Determinant.hpp>
#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>


namespace DO { namespace Sara {

  /*!
    @addtogroup InterestPoint
    @{
  */

  //! Computes a pyramid of determinant of Hessian from the Gaussian pyramid.
  template <typename T>
  ImagePyramid<T> det_of_hessian_pyramid(const ImagePyramid<T>& gaussians)
  {
    ImagePyramid<T> D;
    D.reset(gaussians.num_octaves(),
            gaussians.num_scales_per_octave(),
            gaussians.scale_initial(),
            gaussians.scale_geometric_factor());

    for (int o = 0; o < D.num_octaves(); ++o)
    {
      D.octave_scaling_factor(o) = gaussians.octave_scaling_factor(o);
      for (int s = 0; s < D.num_scales_per_octave(); ++s)
      {
        D(s, o) = gaussians(s, o)
                      .template compute<Hessian>()
                      .template compute<Determinant>();
        // Apply the scale normalization.
        D(s, o).flat_array() *=
            std::pow(gaussians.scale_relative_to_octave(s), 4);
      }
    }
    return D;
  }

  //! Functor class to compute Hessian-Laplace maxima.
  class DO_SARA_EXPORT ComputeHessianLaplaceMaxima
  {
  public:
    /*!
      @brief Constructor
      @param[in]
        extremum_thres
        the response threshold which local maxima of the determinant of Hessian
        function must satisfy.
      @param[in]
        img_padding_sz
        This variable indicates the minimum border size of the image.
        Maxima of determinant of Hessians located in the border of width
        'img_padding_sz' are discarded.
      @param[in]
        numScales
        This variable indicates the number of scales to search in order to
        select the characteristic scale of a corner.
      @param[in]
        extremumRefinementIter
        This variable controls the number of iterations to refine the
        localization of DoG extrema in scale-space. The refinement process is
        based on the function **DO::refineExtremum()**.
     */
    ComputeHessianLaplaceMaxima(
        const ImagePyramidParams& pyrParams = ImagePyramidParams(-1, 3 + 1),
        float extremum_thres = 1e-5f, int img_padding_sz = 1,
        int numScales = 10, int extremumRefinementIter = 5)
      : _pyr_params(pyrParams)
      , _extremum_thres(extremum_thres)
      , _img_padding_sz(img_padding_sz)
      , _extremum_refinement_iter(extremumRefinementIter)
      , _num_scales(numScales)
    {
    }

    /*!
      @brief Localizes Hessian-Laplace maxima for a given image.

      This function does the following:
      1. Constructs a gaussian pyramid \f$\nabla g_{\sigma(s,o)} * I\f$ from
      the image \f$I\f$, where \f$(s,o)\f$ are integers. Here:
      \f$\sigma(s,o) = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per
      octaves.

      2. Localize maxima in determinant of Hessians functions in each scale
         \f$\sigma(s,o)\f$.

      \param[in, out] scale_octave_pairs a pointer to vector of scale and octave
      index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the determinant
      of Hessians.

      \return set of Hessian-Laplace maxima in **std::vector<OERegion>** in each
      scale-normalized determinant of Hessians.
     */
    std::vector<OERegion> operator()(const ImageView<float>& I,
                                     std::vector<Point2i> *scale_octave_pairs = 0);

    /*!
      @brief Returns the Gaussian pyramid used to select characteristic scales
      for Hessian-Laplace interest points.

      The Gaussian pyramid is available after calling the function method
      **ComputeHessianLaplaceExtrema::operator()(I, scale_octave_pairs)** for
      the given image **I**.

      \return the Gaussian pyramid used to localize Hessian-Laplace extrema
      of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    {
      return _gaussians;
    }

    /*!
      @brief Returns the pyramid of Hessian-Laplace functions used to localize
      scale-space extrema of image **I**.

      The pyramid of determinant of Hessians is available after calling the
      function method
      **ComputeHessianLaplaceExtrema::operator()(I, scale_octave_pairs)**,

      \return the pyramid of determinant of Hessians used to localize
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& det_of_hessians() const
    {
      return _det_hessians;
    }

  private: /* data members. */
    //! @{
    //! Parameters
    ImagePyramidParams _pyr_params;
    float _extremum_thres;
    int _img_padding_sz;
    int _extremum_refinement_iter;
    int _num_scales;
    //! @}

    //! @{
    //! Difference of Gaussians.
    ImagePyramid<float> _gaussians;
    ImagePyramid<float> _det_hessians;
    //! @}
  };

  //! Functor class to compute local extrema of determinant of Hessians
  //! in scale space.
  class DO_SARA_EXPORT ComputeDoHExtrema
  {
  public:
    /*!
      @brief Constructor
      @param[in]
        extremum_thres
        the response threshold which local maxima of the determinant of Hessian
        function must satisfy.
      @param[in]
        img_padding_sz
        This variable indicates the minimum border size of the image.
        Maxima of determinant of Hessians located in the border of width
        'img_padding_sz' are discarded.
      @param[in]
        numScales
        This variable indicates the number of scales to search in order to
        select the characteristic scale of a corner.
      @param[in]
        extremumRefinementIter
        This variable controls the number of iterations to refine the
        localization of DoG extrema in scale-space. The refinement process is
        based on the function **DO::refineExtremum()**.
     */
    ComputeDoHExtrema(const ImagePyramidParams& pyrParams =
                          ImagePyramidParams(-1, 3 + 2, pow(2.f, 1.f / 3.f), 2),
                      float extremum_thres = 1e-6f,
                      float edge_ratio_thres = 10.f,
                      int img_padding_sz = 1,
                      int extremum_refinement_iter = 2)
      : pyr_params_(pyrParams)
      , _extremum_thres(extremum_thres)
      , _edge_ratio_thres(edge_ratio_thres)
      , _img_padding_sz(img_padding_sz)
      , _extremum_refinement_iter(extremum_refinement_iter)
    {
    }

    /*!
      @brief Localizes scale-space extrema of determinant of Hessians for a
      given image.

      This function does the following:
      1. Constructs a gaussian pyramid \f$\nabla g_{\sigma(s,o)} * I\f$ from
      the image \f$I\f$, where \f$(s,o)\f$ are integers. Here:
      \f$\sigma(s,o) = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per
      octaves.

      2. Localize extrema in determinant of Hessians functions in each scale
         \f$\sigma(s,o)\f$.

      \param[in, out] scale_octave_pairs a pointer to vector of scale and octave
      index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the determinant
      of Hessians.

      \return set of DoHs extrema in **std::vector<OERegion>** in each
      scale-normalized determinant of Hessians.
     */
    std::vector<OERegion> operator()(const ImageView<float>& I,
                                     std::vector<Point2i> *scale_octave_pairs = 0);

    /*!
      @brief Returns the Gaussian pyramid used to compute DoH extrema.

      The Gaussian pyramid is available after calling the function method
      **ComputeHessianLaplaceExtrema::operator()(I, scale_octave_pairs)** for
      the given image **I**.

      \return the Gaussian pyramid used to localize DoH extrema of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    {
      return _gaussians;
    }

    /*!
      @brief Returns the pyramid of determinant of Hessians used to localize
      scale-space extrema of image **I**.

      The pyramid of determinant of Hessians is available after calling the
      function method
      **ComputeHessianLaplaceExtrema::operator()(I, scale_octave_pairs)**,

      \return the pyramid of determinant of Hessians used to localize
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& det_of_hessians() const
    {
      return _det_hessians;
    }

  private: /* data members. */
    // Parameters
    ImagePyramidParams pyr_params_;
    float _extremum_thres;
    float _edge_ratio_thres;
    int _img_padding_sz;
    int _extremum_refinement_iter;
    // Difference of Gaussians.
    ImagePyramid<float> _gaussians;
    ImagePyramid<float> _det_hessians;
  };

  //! @}


} /* namespace Sara */
} /* namespace DO */
