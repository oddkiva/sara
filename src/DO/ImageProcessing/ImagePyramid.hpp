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
    double cameraSigma() const { return camera_sigma_; }
    /*!
      The image pyramid discretizes image function 
      \f$(x,y,\sigma) \mapsto I(x,y,\sigma)\f$ in the scale-space.
      With the discretization, the image pyramid will consist of a stack of 
      blurred images \f$(I_{\sigma_i})_{1 \leq i \leq N}\f$.

      Here initSigma() corresponds to \f$\sigma_0\f$.
     */
    double initSigma() const { return init_sigma_; }
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
    double scaleGeomFactor() const { return scale_geom_factor_; }
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
    int numScalesPerOctave() const { return num_scales_per_octave_; }
    //! This controls the maximum number of octaves.
    int imagePaddingSize() const { return image_padding_size_; }
    /*!
      \todo. Improve explanation.
      \f$(1/2)^i\f$ is the rescaling factor of the downsampled image of octave 
      \f$i\f$.
     */
    int initOctaveIndex() const { return init_octave_index_; }
    
    ImagePyramidParams(int init_octave_index = -1,
                       int num_scales_per_octave = 3+3,
                       double scale_geom_factor = std::pow(2., 1./3.),
                       int image_padding_size = 1,
                       double camera_sigma = 0.5,
                       double init_sigma = 1.6)
    {
      camera_sigma_ = camera_sigma;
      init_sigma_ = init_sigma;
      num_scales_per_octave_ = num_scales_per_octave;
      scale_geom_factor_ = scale_geom_factor;
      image_padding_size_ = image_padding_size;
      init_octave_index_ = init_octave_index;
    }

  private:
    double camera_sigma_;
    double init_sigma_;
    int num_scales_per_octave_;
    double scale_geom_factor_;
    int image_padding_size_;
    int init_octave_index_;
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
    typedef std::vector<Image<Color, N> > Octave;
    typedef typename ColorTraits<Color>::ChannelType Scalar;
    // Constructor
    inline ImagePyramid() {}
    // Reset image pyramid with the following parameters.
    void reset(int numOctaves,
               int numScalesPerOctave,
               Scalar initSigma,
               Scalar scaleGeomFactor)
    {
      octaves_.clear();
      oct_scaling_factors_.clear();

      octaves_.resize(numOctaves);
      oct_scaling_factors_.resize(numOctaves);
      for (int o = 0; o < numOctaves; ++o)
        octaves_[o].resize(numScalesPerOctave);
      
      init_sigma_ = initSigma;
      scale_geom_factor_ = scaleGeomFactor;
    }
    // Mutable accessors
    Octave& operator()(int o)
    { return octaves_[o]; }
    Image<Color, N>& operator()(int s, int o)
    { return octaves_[o][s]; }
    Color& operator()(int x, int y, int s, int o)
    { return octaves_[o][s](x,y); }
    Scalar& octaveScalingFactor(int o)
    { return oct_scaling_factors_[o]; }
    // Constant accessors.
    const Octave& operator()(int o) const
    { return octaves_[o]; }
    const Image<Color, N>& operator()(int s, int o) const
    { return octaves_[o][s]; }
    const Color& operator()(int x, int y, int s, int o) const
    { return octaves_[o][s](x,y); }
    Scalar octaveScalingFactor(int o) const
    { return oct_scaling_factors_[o]; }
    // Scale and smoothing query.
    int numOctaves() const
    { return static_cast<int>(octaves_.size()); }
    int numScalesPerOctave() const
    { return static_cast<int>(octaves_.front().size()); }
    Scalar initScale() const
    { return init_sigma_; }
    Scalar scaleGeomFactor() const
    { return scale_geom_factor_; }
    Scalar octRelScale(int s) const
    { return pow(scale_geom_factor_, s)*init_sigma_; }
    Scalar scale(int s, int o) const
    { return oct_scaling_factors_[o]*octRelScale(s); }
  
  protected: /* data members */
    Scalar init_sigma_;
    Scalar scale_geom_factor_;
    std::vector<Octave> octaves_;
    std::vector<Scalar> oct_scaling_factors_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_IMAGEPYRAMID_HPP */
