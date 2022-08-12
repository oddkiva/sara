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

#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  /*!
   *  @ingroup ImageProcessing
   *  @defgroup ScaleSpace Scale-Space Representation
   *  @{
   */


  //! Image pyramid parameters used to discretize the Gaussian scale-space.
  class ImagePyramidParams
  {
  public:
    //! @brief Constructor.
    ImagePyramidParams(                                                 //
        const int first_octave_index = -1,                              //
        const int scale_count_per_octave = 3 + 3,                       //
        const float scale_geometric_factor = std::pow(2.f, 1.f / 3.f),  //
        const int image_padding_size = 1,                               //
        const float scale_camera = 0.5f,                                //
        const float scale_initial = 1.6f,                               //
        const int num_octaves_max = std::numeric_limits<int>::max())
    {
      _scale_camera = scale_camera;
      _scale_initial = scale_initial;
      _scale_count_per_octave = scale_count_per_octave;
      _scale_geometric_factor = scale_geometric_factor;
      _image_padding_size = image_padding_size;
      _first_octave_index = first_octave_index;
      _num_octaves_max = num_octaves_max;
    }

    /*!
     *  Let @f$I@f$ denote an image function that maps a position @f$(x, y) \in
     *  \mathbb{R}^2@f$ to an intensity value in @f$\mathbb{R}@f$.
     *
     *  In theory, this corresponds to an ideal image function where we know the
     *  intensity value with perfect precision at any position @f$(x,
     *  y)@f$ at any fine resolution @f$\sigma@f$ in meters (millimeters,
     *  microns, and so on...).
     *
     *  The resolution is termed as scale.
     *
     *  There is this nice stackexchange answer:
     *  https://biology.stackexchange.com/questions/26189/whats-the-smallest-size-a-human-eye-can-see.
     *
     *  The human eye can perceive and distinguish small objects up to the order
     *  of tens of microns, so the human eye photoreceptor has a predefined
     *  size of a few microns. This is the scale of the image.
     *
     *  Likewise the camera will capture photographs with details at a fixed
     *  resolution @f$\sigma@f$ also defined by the size of its photoreceptors.
     *  The photoreceptor can be thought as a tiny square of size 1x1 pixel. At
     *  this resolution and at coarser resolutions, the average pixel value can
     *  be known with good certainty.
     *
     *  Each pixel value of a photograph can be viewed as a local averaging of
     *  all color values @f$I(x, y)@f$ on this photoreceptor square, which is
     *  less than a millimeter.
     *
     *  In the scale-space framework, the local averaging is modeled by a
     *  Gaussian convolution and the image function is augmented with
     *  an additional dimension which is the scale:
     *
     *  @f$ I : (x,y,\sigma) \mapsto (g_\sigma * I)\ (x,y) @f$,
     *
     *  where @f$g_\sigma@f$ is the Gaussian distribution with zero mean and
     *  standard deviation @f$\sigma@f$.
     *
     *  We also denote by @f$ I_\sigma : (x,y) \mapsto I(x,y,\sigma) @f$ the
     *  image @f$I@f$ at scale @f$\sigma@f$.
     *
     *  By design, the camera samples the perfect image @f$I_0@f$ at a
     *  predefined scale and the image acquired by the camera is denoted as
     *  @f$I_{\sigma_\textrm{camera}}@f$.
     *
     *  Here scale_camera() corresponds to camera resolution
     *  @f$\sigma_\textrm{camera}@f$.
     *
     *  The digital image that computers read is
     *  @f$I_{\sigma_\textrm{camera}}@f$ and not @f$I@f$.
     *
     *  Now the caveat here is that @f$\sigma_\textrm{camera}@f$ is not known in
     *  meters and we work with pixels. If we know the camera parameters, we can
     *  relate the size of a pixel in millimeters.
     *
     *  The default parameter we use is @f$0.5@f$, which means that we assume we
     *  can estimate the pixel values with good confidence up to half a pixel
     *  resolution.
     *
     *  We can retrieve such color values every half pixels by using bilinear
     *  interpolation.
     */
    float scale_camera() const
    {
      return _scale_camera;
    }

    /*!
     *  The image pyramid discretizes image function
     *  @f$(x,y,\sigma) \mapsto I(x,y,\sigma)@f$ in the scale-space.
     *  With the discretization, the image pyramid will consist of a stack of
     *  blurred images @f$(I_{\sigma_i})_{0 \leq i < N}@f$.
     *
     *  Here scale_initial() corresponds to @f$\sigma_0@f$.
     */
    float scale_initial() const
    {
      return _scale_initial;
    }

    /*!
     *  The sequence @f$ (\sigma_i)_{1 \leq i \leq N} @f$ follows a geometric
     *  progression, i.e., @f$\sigma_i = k^i \sigma_0@f$.
     *
     *  Laplacians of Gaussians @f$ \nabla^2 I_{\sigma} @f$ can be approximated
     *  efficiently with differences of Gaussians @f$ I_{k\sigma} - I_{\sigma}
     * @f$ for each @f$ \sigma = \sigma_i @f$ without the need to normalize the
     *  difference of Gaussians.
     *
     *  Thus
     *  \f{eqnarray*}{
     *    \frac{\partial I_\sigma}{\partial \sigma} &=& \sigma \nabla^2 I_\sigma
     * \\ I_{k\sigma} - I_{\sigma} &\approx& (k-1) \sigma^2 \nabla^2 I_\sigma
     *  \f}
     *
     *  Here scale_geometric_factor() corresponds to the value @f$ k @f$.
     */
    float scale_geometric_factor() const
    {
      return _scale_geometric_factor;
    }

    /*!
     *  The smoothed image @f$I_{2\sigma}@f$ is equivalent to the downsampled
     *  image @f$I_{\sigma}@f$. Because of this observation, a pyramid of
     *  gaussians is divided into octaves.
     *
     *  We call an octave a stack of @f$S@f$ blurred images @f$I_{\sigma_i}@f$
     *  separated by a constant factor @f$k@f$ in the scale space, i.e.,
     *  @f$ \sigma_{i+1} = k \sigma_i @f$.
     *  The number of scales in each octave is the integer @f$S@f$ such that
     *  @f$k^S \sigma = 2 \sigma@f$, i.e., @f$ k= 2^{1/S} @f$.
     */
    int scale_count_per_octave() const
    {
      return _scale_count_per_octave;
    }

    //! @brief This impacts the maximum number of octaves.
    int image_padding_size() const
    {
      return _image_padding_size;
    }

    /*!
     *  @todo. Improve explanation.
     *  @f$(1/2)^i@f$ is the rescaling factor of the downsampled image of octave
     *  @f$i@f$.
     */
    int first_octave_index() const
    {
      return _first_octave_index;
    }

    //! @brief The maximum number of octaves to generate.
    int num_octaves_max() const
    {
      return _num_octaves_max;
    }

  private:
    float _scale_camera;
    float _scale_initial;
    int _scale_count_per_octave;
    float _scale_geometric_factor;
    int _image_padding_size;
    int _first_octave_index;
    int _num_octaves_max;
  };


  /*!
    The image pyramid is regular, i.e., it has:
    - the same number of scales in each octave
    - the same geometric progression factor in the scale in each octave
   */
  template <typename Pixel, int N = 2>
  class ImagePyramid
  {
  public: /* member functions */
    //! Convenient typedefs.
    //! @{
    using pixel_type = Pixel;
    using image_type = Image<Pixel>;
    using octave_type = std::vector<Image<Pixel, N>>;
    using scalar_type = typename PixelTraits<Pixel>::channel_type;
    //! @}

    //! @brief Default constructor.
    inline ImagePyramid() = default;

    //! @brief Reset image pyramid with the following parameters.
    void reset(int num_octaves, int num_scales_per_octave, float scale_initial,
               float scale_geometric_factor)
    {
      _octaves.clear();
      _oct_scaling_factors.clear();

      _octaves.resize(num_octaves);
      _oct_scaling_factors.resize(num_octaves);
      for (int o = 0; o < num_octaves; ++o)
        _octaves[o].resize(num_scales_per_octave);

      _scale_initial = scale_initial;
      _scale_geometric_factor = scale_geometric_factor;
    }

    //! @brief Mutable octave getter.
    octave_type& operator()(int o)
    {
      return _octaves[o];
    }

    //! @brief Mutable image getter.
    image_type& operator()(int s, int o)
    {
      return _octaves[o][s];
    }

    //! @brief Mutable pixel getter.
    pixel_type& operator()(int x, int y, int s, int o)
    {
      return _octaves[o][s](x, y);
    }

    //! @brief Mutable getter of the octave scaling factor.
    auto octave_scaling_factor(int o) -> float&
    {
      return _oct_scaling_factors[o];
    }

    //! @brief Immutable octave getter.
    auto operator()(int o) const -> const octave_type&
    {
      return _octaves[o];
    }

    //! @brief Immutable image getter.
    auto operator()(int s, int o) const -> const image_type&
    {
      return _octaves[o][s];
    }

    //! @brief Immutable pixel getter.
    auto operator()(int x, int y, int s, int o) const -> const pixel_type&
    {
      return _octaves[o][s](x, y);
    }

    //! @brief Immutable getter of the octave scaling factor.
    auto octave_scaling_factor(int o) const -> float
    {
      return _oct_scaling_factors[o];
    }

    //! @brief Immutable getter of the number of octaves.
    auto octave_count() const
    {
      return static_cast<int>(_octaves.size());
    }

    //! @brief Immutable getter of the number of scales per octave.
    auto scale_count_per_octave() const
    {
      return static_cast<int>(_octaves.front().size());
    }

    //! @brief Returns the number of scales (across all octaves).
    auto scale_count() const
    {
      return octave_count() * scale_count_per_octave();
    }

    //! @brief Immutable getter of the initial scale.
    auto scale_initial() const
    {
      return _scale_initial;
    }

    //! @brief Immutable getter of the scale geometric factor.
    auto scale_geometric_factor() const
    {
      return _scale_geometric_factor;
    }

    //! @brief Immutable getter of the relative scale w.r.t. an octave.
    auto scale_relative_to_octave(int s) const
    {
      return std::pow(_scale_geometric_factor, s) * _scale_initial;
    }

    //! @brief Immutable getter of the scale relative to an octave.

    auto scale(int s, int o) const
    {
      return _oct_scaling_factors[o] * scale_relative_to_octave(s);
    }

  protected: /* data members */
    //! @{
    //! @brief Parameters.
    float _scale_initial;
    float _scale_geometric_factor;
    //! @}

    //! @{
    //! @brief Image data.
    std::vector<octave_type> _octaves;
    std::vector<float> _oct_scaling_factors;
    //! @}
  };

  //! @}

}}  // namespace DO::Sara
