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
  
  //! Functor class to compute Hessian-Laplace maxima.
  class ComputeHessianLaplaceMaxima
  {
  public:
    /*!
      \brief Constructor
      @param[in]
        extremumThres
        the response threshold which local maxima of the determinant of Hessian 
        function must satisfy.
      @param[in]
        imgPaddingSz
        This variable indicates the minimum border size of the image.
        Maxima of determinant of Hessians located in the border of width 
        'imgPaddingSz' are discarded.
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
    ComputeHessianLaplaceMaxima(const ImagePyramidParams& pyrParams = 
                                  ImagePyramidParams(-1, 3+1),
                                float extremumThres = 1e-5f,
                                int imgPaddingSz = 1,
                                int numScales = 10,
                                int extremumRefinementIter = 5)
      : pyr_params_(pyrParams)
      , extremum_thres_(extremumThres)
      , img_padding_sz_(imgPaddingSz)
      , extremum_refinement_iter_(extremumRefinementIter)
      , num_scales_(numScales)
    {}
    /*!
      \brief Localizes Hessian-Laplace maxima for a given image.

      This function does the following:
      1. Constructs a gaussian pyramid \f$\nabla g_{\sigma(s,o)} * I\f$ from 
      the image \f$I\f$, where \f$(s,o)\f$ are integers. Here:
      \f$\sigma(s,o) = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per
      octaves.

      2. Localize maxima in determinant of Hessians functions in each scale
         \f$\sigma(s,o)\f$.
      
      \param[in, out] scaleOctavePairs a pointer to vector of scale and octave
      index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the determinant 
      of Hessians.

      \return set of Hessian-Laplace maxima in **std::vector<OERegion>** in each
      scale-normalized determinant of Hessians.
     */
    std::vector<OERegion> operator()(const Image<float>& I,
                                     std::vector<Point2i> *scaleOctavePairs = 0);
    /*!
      \brief Returns the Gaussian pyramid used to select characteristic scales
      for Hessian-Laplace interest points.
      
      The Gaussian pyramid is available after calling the function method
      **ComputeHessianLaplaceExtrema::operator()(I, scaleOctavePairs)** for 
      the given image **I**.

      \return the Gaussian pyramid used to localize Hessian-Laplace extrema
      of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    { return gaussians_; }
    /*!
      \brief Returns the pyramid of Hessian-Laplace functions used to localize 
      scale-space extrema of image **I**.

      The pyramid of determinant of Hessians is available after calling the 
      function method 
      **ComputeHessianLaplaceExtrema::operator()(I, scaleOctavePairs)**, 
      
      \return the pyramid of determinant of Hessians used to localize 
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& detOfHessians() const
    { return det_hessians_; }
  private: /* data members. */
    // Parameters
    ImagePyramidParams pyr_params_;
    float extremum_thres_;
    int img_padding_sz_;
    int extremum_refinement_iter_;
    int num_scales_;
    // Difference of Gaussians.
    ImagePyramid<float> gaussians_;
    ImagePyramid<float> det_hessians_;
  };

  //! Functor class to compute local extrema of determinant of Hessians 
  //! in scale space.
  class ComputeDoHExtrema
  {
  public:
    /*!
      \brief Constructor
      @param[in]
        extremumThres
        the response threshold which local maxima of the determinant of Hessian 
        function must satisfy.
      @param[in]
        imgPaddingSz
        This variable indicates the minimum border size of the image.
        Maxima of determinant of Hessians located in the border of width 
        'imgPaddingSz' are discarded.
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
                        ImagePyramidParams(-1, 3+2, pow(2.f, 1.f/3.f), 2),
                      float extremumThres = 1e-6f,
                      float edgeRatioThres = 10.f,
                      int imgPaddingSz = 1,
                      int extremumRefinementIter = 2)
      : pyr_params_(pyrParams)
      , extremum_thres_(extremumThres)
      , edge_ratio_thres_(edgeRatioThres)
      , img_padding_sz_(imgPaddingSz)
      , extremum_refinement_iter_(extremumRefinementIter)
    {}
    /*!
      \brief Localizes scale-space extrema of determinant of Hessians for a 
      given image.

      This function does the following:
      1. Constructs a gaussian pyramid \f$\nabla g_{\sigma(s,o)} * I\f$ from 
      the image \f$I\f$, where \f$(s,o)\f$ are integers. Here:
      \f$\sigma(s,o) = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per
      octaves.

      2. Localize extrema in determinant of Hessians functions in each scale
         \f$\sigma(s,o)\f$.
      
      \param[in, out] scaleOctavePairs a pointer to vector of scale and octave
      index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the determinant 
      of Hessians.

      \return set of DoHs extrema in **std::vector<OERegion>** in each
      scale-normalized determinant of Hessians.
     */
    std::vector<OERegion> operator()(const Image<float>& I,
                                     std::vector<Point2i> *scaleOctavePairs = 0);
    /*!
      \brief Returns the Gaussian pyramid used to compute DoH extrema.
      
      The Gaussian pyramid is available after calling the function method
      **ComputeHessianLaplaceExtrema::operator()(I, scaleOctavePairs)** for 
      the given image **I**.

      \return the Gaussian pyramid used to localize DoH extrema of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    { return gaussians_; }
    /*!
      \brief Returns the pyramid of determinant of Hessians used to localize 
      scale-space extrema of image **I**.

      The pyramid of determinant of Hessians is available after calling the 
      function method 
      **ComputeHessianLaplaceExtrema::operator()(I, scaleOctavePairs)**, 
      
      \return the pyramid of determinant of Hessians used to localize 
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& detOfHessians() const
    { return det_hessians_; }
  private: /* data members. */
    // Parameters
    ImagePyramidParams pyr_params_;
    float extremum_thres_;
    float edge_ratio_thres_;
    int img_padding_sz_;
    int extremum_refinement_iter_;
    // Difference of Gaussians.
    ImagePyramid<float> gaussians_;
    ImagePyramid<float> det_hessians_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDETECTORS_HESSIAN_HPP */
