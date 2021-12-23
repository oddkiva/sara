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


namespace DO::Sara {

  /*!
   *  @ingroup FeatureDetectors
   *  @defgroup InterestPoint Interest Point Detection
   *  @{
   */

  //! Functor class to compute DoG extrema
  class ComputeDoGExtrema
  {
  public:
    //! @brief Constructor
    /*!
     *  @param[in]
     *    extremum_thres
     *    the response threshold which the DoG extremum absolute value
     *    @f$
     *      \left|
     *        \left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I
     *      \right| (\mathbf{x})
     *    @f$
     *    must exceed.
     *    Note that \f$ \sigma(s',o') = 2^{s'/S+o'}\f$  where \f$S\f$ is the
     *    number of scales per octave.
     *  @param[in]
     *    edge_ratio_thres
     *    the Hessian matrix \f$\mathbf{H}\f$ at the local scale-space extremum
     *    must satisfy
     *    @f[
     *      \frac{\mathrm{det}(\mathbf{H})}{\mathrm{tr}(\mathbf{H})} >
     *      \frac{(r+1)^2}{r}
     *    @f]
     *    where \f$r\f$ corresponds to the variable **edge_ratio_thres**.
     *    In terms of implementation, we use the function
     * **DO::Sara::on_edge()**. We use the \f$r=10\f$ as stated in [Lowe, IJCV
     * 2004].
     *  @param[in]
     *    img_padding_sz
     *    This variable indicates the minimum border size of the image. DoG
     *    extrema located at the size-specified border are discarded.
     *  @param[in]
     *    extremum_refinement_iter
     *    This variable controls the number of iterations to refine the
     *    localization of DoG extrema in scale-space. The refinement process is
     *    based on the function **DO::refineExtremum()**.
     */
    inline ComputeDoGExtrema(
        const ImagePyramidParams& pyramid_params = ImagePyramidParams(),
        float extremum_thres = 0.01f, float edge_ratio_thres = 10.f,
        int img_padding_sz = 1, int extremum_refinement_iter = 5)
      : _pyramid_params(pyramid_params)
      , _extremum_thres(extremum_thres)
      , _edge_ratio_thres(edge_ratio_thres)
      , _img_padding_sz(img_padding_sz)
      , _extremum_refinement_iter(extremum_refinement_iter)
    {
      if (_pyramid_params.scale_count_per_octave() < 4)
        throw std::runtime_error{
            "Error: The extraction of DoG extrema needs (1 + 3) = 4 scales per "
            "octave at the very minimum!"};
    }

    //! @brief Localizes DoG extrema for a given image.
    /*!
     *  This function does the following:
     *  1. Constructs a gaussian pyramid @f$\nabla g_{\sigma(s,o)} * I@f$ from
     *  the image @f$I@f$, where \f$(s,o)\f$ are integers. Here:
     *  @f$\sigma(s,o) = 2^{\frac{s}{S} + o}@f$ where @f$S@f$ is the number of
     *  scales per octaves.

     *  2. Localize extrema in each difference of Gaussians
     *  \f$\left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I \f$
     *  where \f$(s,o)\f$ are scale and octave indices.

     *  \param[in, out] scale_octave_pairs a pointer to vector of scale and
     octave
     *  index pairs \f$(s_i,o_i)\f$. This index pair corresponds to the
     difference
     *  of Gaussians
     *  \f$\left( g_{\sigma(s_i+1,o_i)} - g_{\sigma(s_i,o_i)} \right) * I\f$
     *  where the extremum \f$(x_i,y_i,\sigma_i)\f$ is detected.

     *  \return set of DoG extrema in **std::vector<OERegion>** in each
     *  difference of Gaussians
     *  \f$\left( g_{\sigma(s+1,o)} - g_{\sigma(s,o)} \right) * I \f$.
     */
    DO_SARA_EXPORT
    std::vector<OERegion>
    operator()(const ImageView<float>& I,
               std::vector<Point2i>* scale_octave_pairs = 0);

    DO_SARA_EXPORT
    auto operator()(const ImageView<float>& I) -> void;

    //! @brief Returns the Gaussian pyramid used to localize scale-space extrema
    //! of image **I**.
    /*!
     * The Gaussian pyramid is available after calling the function method
     * **ComputeDoGExtrema::operator()(I, scale_octave_pairs)** for the given
     * image **I**.

     * \return the Gaussian pyramid used to localize scale-space extrema
     * of image **I**.
     */
    inline auto gaussians() const -> const ImagePyramid<float>&
    {
      return _gaussians;
    }

    //! @brief Returns the pyramid of difference of Gaussians used to localize
    //! scale-space extrema of image **I**.
    /*!
     *  The pyramid of difference of Gaussians is available after calling the
     *  function method **ComputeDoGExtrema::operator()(I,
     scale_octave_pairs)**,

     *  \return the pyramid of difference of Gaussians used to localize
     *  scale-space extrema of image **I**.
     */
    inline auto diff_of_gaussians() const -> const ImagePyramid<float>&
    {
      return _diff_of_gaussians;
    }

    //! @brief The list of local scale-space extrema at each index
    //!   (scale index = s, octave index = o).
    /*!
     *  By design the lists of scale-space extrema at:
     *  - scale index = 0
     *  - scale index = scale_count - 1
     *  are always empty in each octave.
     *  From the scale index >= 1, the list of scale is non-empty in general.
     */
    inline auto extrema(int s, int o) const -> const std::vector<OERegion>&
    {
      return _extrema[o * _diff_of_gaussians.scale_count_per_octave() + s];
    }

  private: /* data members. */
    //! Parameters
    //! @{
    ImagePyramidParams _pyramid_params;
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

    //! @brief  The sparse list of extrema per scales.
    //!
    //! N.B.: at each index i of the array, we retrieve the list of extrema at
    //the ! following (octave-relative scale) index pair where:
    //! - The octave is: i / num_scales
    //! - The relative scale is: i % num_scales
    std::vector<std::vector<OERegion>> _extrema;
    //! @brief
  };


}  // namespace DO::Sara
