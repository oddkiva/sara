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

#ifndef DO_SARA_FEATUREDETECTORS_HARRIS_HPP
#define DO_SARA_FEATUREDETECTORS_HARRIS_HPP


namespace DO { namespace Sara {

  /*!
    \ingroup InterestPoint
    @{
  */

  // ======================================================================== //
  // Multiscale Harris Corner function.
  /*!
    Computes the Harris-Stephens corner function
      \f$det(\mathbf{A}) - \kappa trace(\mathbf{A})^2\f$
    where \f$\mathbf{A}\f$ is the auto-correlation matrix.
    Good values of \f$\kappa\f$ range in \f$[0.04, 0.15]\f$.

    As a reminder, the auto-correlation matrix \f$A\f$ is defined as below.
    Let \f$g_{\sigma}\f$ be the Gaussian kernel with standard deviation
    \f$\sigma\f$.
    And define the convolved image by \f$I_\sigma = g_{\sigma_I} * I\f$, then
    \f[
      \mathbf{A} =
      g_{\sigma_D} * \left[ \nabla I_{\sigma_I} \nabla I_{\sigma_I}^T \right]
    \f].
   */
  Image<float> scaleAdaptedHarrisCornerness(const Image<float>& I,
                                            float sigma_I, float sigma_D,
                                            float kappa);
  //! Pyramid of Harris cornerness functions.
  ImagePyramid<float> harrisCornernessPyramid(
    const Image<float>& image,
    float kappa = 0.04f,
    const ImagePyramidParams& params = ImagePyramidParams(-1, 2+1, sqrt(2.f), 1));

  // ======================================================================== //
  /*!
    Finds and stores local maxima in the image \f$I\f$ exceeding the specified
    threshold.
   */
  std::vector<OERegion> localMaxima(const Image<float>& I, float thres);
  /*!
    \brief Utility functions to locate edgels from the Harris-Stephens corner
    function.

    A pixel \f$(x,y)\f$ is an edgel if its Harris-Stephens cornerness
    is negative and attains a minimum either in the x-direction or the
    y-direction.
   */
  bool localMinX(int x, int y, Image<float>& I);
  /*!
    \brief Utility functions to locate edgels from the Harris-Stephens corner
    function.

    A pixel \f$(x,y)\f$ is an edgel if its Harris-Stephens cornerness
    is negative and attains a minimum either in the x-direction or the
    y-direction.
   */
  bool localMinY(int x, int y, Image<float>& I);

  //! Functor class to compute Harris-Laplace corners.
  class ComputeHarrisLaplaceCorners
  {
  public:
    /*!
      \brief Constructor
      @param[in]
        kappa
        the user parameter in the Harris corner function. Good values of
        \f$\kappa\f$ range in \f$[0.04, 0.15]\f$. See description of
        function **scaleAdaptedHarrisCornerness** for details
      @param[in]
        extremumThres
        the response threshold which local maxima of the Harris cornerness
        function must satisfy.
      @param[in]
        imgPaddingSz
        This variable indicates the minimum border size of the image. DoG
        extrema located at the size-specified border are discarded.
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
    ComputeHarrisLaplaceCorners(const ImagePyramidParams& pyrParams =
                                  ImagePyramidParams(-1, 2+1, sqrt(2.f), 1),
                                float kappa = 0.04f,
                                float extremumThres = 1e-6f,
                                int imgPaddingSz = 1,
                                int numScales = 10,
                                int extremumRefinementIter = 5)
      : pyr_params_(pyrParams)
      , kappa_(kappa)
      , extremum_thres_(extremumThres)
      , img_padding_sz_(imgPaddingSz)
      , extremum_refinement_iter_(extremumRefinementIter)
      , num_scales_(numScales)
    {}
    /*!
      \brief Localizes Harris cornerness local maxima for a given image.

      This function does the following:
      1. Constructs a gaussian pyramid \f$\nabla g_{\sigma(s,o)} * I\f$ from
      the image \f$I\f$, where \f$(s,o)\f$ are integers. Here:
      \f$\sigma(s,o) = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per
      octaves.

      2. Localize extrema in Harris corner functions in each scale
         \f$\sigma(s,o)\f$.

      \param[in, out] scaleOctavePairs a pointer to vector of scale and octave
      index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the difference
      of Gaussians
      \f$\left( g_{\sigma(s_i+1,o_i)} - g_{\sigma(s_i,o_i)} \right) * I\f$
      where the extremum \f$(x_i,y_i,\sigma_i)\f$ is detected.

      \return set of Harris corners in **std::vector<OERegion>** in each
      difference of Gaussians
      \f$\left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I \f$.
     */
    std::vector<OERegion> operator()(const Image<float>& I,
                                     std::vector<Point2i> *scaleOctavePairs = 0);
    /*!
      \brief Returns the Gaussian pyramid used to select characteristic scales
      for Harris-Laplace corners.

      The Gaussian pyramid is available after calling the function method
      **ComputeHarrisLaplaceExtrema::operator()(I, scaleOctavePairs)** for
      the given image **I**.

      \return the Gaussian pyramid used to localize scale-space extrema
      of image **I**.
     */
    const ImagePyramid<float>& gaussians() const
    { return gaussians_; }
    /*!
      \brief Returns the pyramid of Harris corner functions used to localize
      scale-space extrema of image **I**.

      The pyramid of Harris corner functions is available after calling the
      function method
      **ComputeHarrisLaplaceExtrema::operator()(I, scaleOctavePairs)**,

      \return the pyramid of difference of Gaussians used to localize
      scale-space extrema of image **I**.
     */
    const ImagePyramid<float>& harris() const
    { return harris_; }
  private: /* data members. */
    // Parameters
    ImagePyramidParams pyr_params_;
    float kappa_;
    float extremum_thres_;
    int img_padding_sz_;
    int extremum_refinement_iter_;
    int num_scales_;
    // Difference of Gaussians.
    ImagePyramid<float> gaussians_;
    ImagePyramid<float> harris_;
  };

  //! @}


} /* namespace Sara */
} /* namespace DO */

#endif /* DO_SARA_FEATUREDETECTORS_HARRIS_HPP */
