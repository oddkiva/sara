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

#ifndef DO_IMAGEPROCESSING_IMAGEPYRAMID_HPP
#define DO_IMAGEPROCESSING_IMAGEPYRAMID_HPP


#include <DO/Core/Image.hpp>


namespace DO {

  /*!
    \ingroup ImageProcessing
    \defgroup ScaleSpace Scale-Space Representation
    @{
   */


  //! Image pyramid parameters which discretizes the Gaussian scale-space.
  class ImagePyramidParams
  {
  public:
    /*!
      Let us consider an image \f$I\f$. In the scale-space framework, we work 
      with the scale-space image function 
      \f$ I : (x,y,\sigma) \mapsto (g_\sigma * I)\ (x,y) \f$,
      where \f$g_\sigma\f$ is the gaussian distribution with zero mean and 
      standard deviation \f$\sigma\f$.

      We also denote by \f$ I_\sigma : (x,y) \mapsto I(x,y,\sigma) \f$ the 
      image \f$I\f$ at scale \f$\sigma\f$.
      Note that the real image is \f$I_0: (x,y) \mapsto I(x,y)\f$.
      Because of the spatial sampling, the camera captures the real image 
      \f$I_0\f$ with some blurring and the camera-acquired image is
      \f$I_{\sigma_\textrm{camera}}\f$.

      Here cameraSigma() thus corresponds to \f$\sigma_\textrm{camera}\f$.
     */
    double sigma_camera() const
    {
      return _sigma_camera;
    }

    /*!
      The image pyramid discretizes image function 
      \f$(x,y,\sigma) \mapsto I(x,y,\sigma)\f$ in the scale-space.
      With the discretization, the image pyramid will consist of a stack of 
      blurred images \f$(I_{\sigma_i})_{1 \leq i \leq N}\f$.

      Here initSigma() corresponds to \f$\sigma_0\f$.
     */
    double sigma_initial() const
    {
      return _sigma_initial;
    }

    /*!
      The sequence \f$ (\sigma_i)_{1 \leq i \leq N} \f$ follows a geometric 
      progression, i.e., \f$\sigma_i = k^i \sigma_0\f$. 

      Laplacians of Gaussians \f$ \nabla^2 I_{\sigma} \f$ can be approximated 
      efficiently with differences of Gaussians \f$ I_{k\sigma} - I_{\sigma} \f$
      for each \f$ \sigma = \sigma_i \f$*without needing to renormalize*. 
      
      Indeed:
      \f{eqnarray*}{
        \frac{\partial I_\sigma}{\partial \sigma} &=& \sigma \nabla^2 I_\sigma \\
        I_{k\sigma} - I_{\sigma} &\approx& (k-1) \sigma^2 \nabla^2 I_\sigma
      \f}

      Here scaleGeomFactor() corresponds to the value \f$ k \f$.
     */
    double scale_geometric_factor() const
    {
      return _scale_geometric_factor;
    }

    /*!
      The smoothed image \f$I_{2\sigma}\f$ is equivalent to the downsampled 
      image \f$I_{\sigma}\f$. Because of this observation, a pyramid of 
      gaussians is divided into octaves.

      We call an octave a stack of \f$S\f$ blurred images \f$I_{\sigma_i}\f$ 
      separated by a constant factor \f$k\f$ in the scale space, i.e.,
      \f$ \sigma_{i+1} = k \sigma_i \f$.
      The number of scales in each octave is the integer \f$S\f$ such that 
      \f$k^S \sigma = 2 \sigma\f$, i.e., \f$ k= 2^{1/S} \f$.
     */
    int num_scales_per_octaves() const
    {
      return _num_scales_per_octave;
    }

    //! This controls the maximum number of octaves.
    int image_padding_size() const
    {
      return _image_padding_size;
    }

    /*!
      \todo. Improve explanation.
      \f$(1/2)^i\f$ is the rescaling factor of the downsampled image of octave 
      \f$i\f$.
     */
    int first_octave_index() const
    {
      return _first_octave_index;
    }

    //! \brief Constructor.
    ImagePyramidParams(int first_octave_index = -1,
                       int num_scales_per_octave = 3+3,
                       double scale_geometric_factor = std::pow(2., 1./3.),
                       int image_padding_size = 1,
                       double camera_sigma = 0.5,
                       double init_sigma = 1.6)
    {
      _sigma_camera = camera_sigma;
      _sigma_initial = init_sigma;
      _num_scales_per_octave = num_scales_per_octave;
      _scale_geometric_factor = scale_geometric_factor;
      _image_padding_size = image_padding_size;
      _first_octave_index = first_octave_index;
    }

  private:
    double _sigma_camera;
    double _sigma_initial;
    int _num_scales_per_octave;
    double _scale_geometric_factor;
    int _image_padding_size;
    int _first_octave_index;
  };


  /*!
    The image pyramid is regular, i.e., it has:
    - the same number of scales in each octave
    - the same geometric progression factor in the scale in each octave
   */
  template <typename Color, int N=2>
  class ImagePyramid
  {
  public: /* member functions */
    // Convenient alias
    typedef std::vector<Image<Color, N> > octave_type;
    typedef typename PixelTraits<Color>::channel_type scalar_type;

    // Constructor
    inline ImagePyramid() {}
    // Reset image pyramid with the following parameters.
    void reset(int num_octaves,
               int num_scales_per_octave,
               scalar_type sigma_initial,
               scalar_type scale_geometric_factor)
    {
      _octaves.clear();
      _oct_scaling_factors.clear();

      _octaves.resize(num_octaves);
      _oct_scaling_factors.resize(num_octaves);
      for (int o = 0; o < num_octaves; ++o)
        _octaves[o].resize(num_scales_per_octave);
      
      _sigma_initial = sigma_initial;
      _scale_geometric_factor = scale_geometric_factor;
    }

    // Mutable accessors
    octave_type& operator()(int o)
    {
      return _octaves[o];
    }

    Image<Color, N>& operator()(int s, int o)
    {
      return _octaves[o][s];
    }

    Color& operator()(int x, int y, int s, int o)
    {
      return _octaves[o][s](x,y);
    }

    scalar_type& octave_scaling_factor(int o)
    {
      return _oct_scaling_factors[o];
    }

    // Constant accessors.
    const octave_type& operator()(int o) const
    {
      return _octaves[o];
    }

    const Image<Color, N>& operator()(int s, int o) const
    {
      return _octaves[o][s];
    }

    const Color& operator()(int x, int y, int s, int o) const
    {
      return _octaves[o][s](x,y);
    }

    scalar_type octave_scaling_factor(int o) const
    {
      return _oct_scaling_factors[o];
    }

    // Scale and smoothing query.
    int num_octaves() const
    {
      return static_cast<int>(_octaves.size());
    }

    int num_scales_per_octave() const
    {
      return static_cast<int>(_octaves.front().size());
    }

    scalar_type scale_initial() const
    {
      return _sigma_initial;
    }

    scalar_type scale_geometric_factor() const
    {
      return _scale_geometric_factor;
    }

    scalar_type relative_scale_to_octave(int s) const
    {
      return pow(_scale_geometric_factor, s)*_sigma_initial;
    }

    scalar_type scale(int s, int o) const
    {
      return _oct_scaling_factors[o]*relative_scale_to_octave(s);
    }

  protected: /* data members */
    scalar_type _sigma_initial;
    scalar_type _scale_geometric_factor;
    std::vector<octave_type> _octaves;
    std::vector<scalar_type> _oct_scaling_factors;
  };

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_IMAGEPYRAMID_HPP */
