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

#ifndef DO_IMAGEPROCESSING_DOG_HPP
#define DO_IMAGEPROCESSING_DOG_HPP

namespace DO {

  /*!
    \ingroup ScaleSpace
    @{
   */

  //! Computes a pyramid of Gaussians.
  template <typename T>
  ImagePyramid<T>
  gaussianPyramid(const Image<T>& image,
                  const ImagePyramidParams& params = ImagePyramidParams())
  {
    typedef typename ImagePyramid<T>::Scalar Scalar;
    // Resize the image with the appropriate factor.
    Scalar resizeFactor = pow(2.f, -params.initOctaveIndex());
    Image<T> I(enlarge(image, resizeFactor) );
    // Deduce the new camera sigma with respect to the dilated image.
    Scalar cameraSigma = Scalar(params.cameraSigma())*resizeFactor;
    // Blur the image so that its new sigma is equal to the initial sigma.
    Scalar initSigma = Scalar(params.initSigma());
    if (cameraSigma < initSigma)
    {
      Scalar sigma = sqrt(initSigma*initSigma - cameraSigma*cameraSigma);
      I = gaussian(I, sigma);
    }

    // Deduce the maximum number of octaves.
    int l = std::min(image.width(), image.height());
    int b = params.imagePaddingSize();
    // l/2^k > 2b
    // 2^k < l/(2b)
    // k < log(l/(2b))/log(2)
    int numOctaves = static_cast<int>(log(l/(2.f*b))/log(2.f));
    
    // Shorten names.
    Scalar k = Scalar(params.scaleGeomFactor());
    int numScales = params.numScalesPerOctave();
    int downscaleIndex = int( floor( log(Scalar(2))/log(k)) );

    // Create the image pyramid
    ImagePyramid<T> G;
    G.reset(numOctaves, numScales, initSigma, k);

    //omp_set_num_threads(1);

    for (int o = 0; o < numOctaves; ++o)
    {
      // Compute the octave scaling factor
      G.octaveScalingFactor(o) = 
        (o == 0) ? 1.f/resizeFactor : G.octaveScalingFactor(o-1)*2;

      // Compute the gaussians in octave \f$o\f$
      Scalar sigma_s_1 = initSigma;
      G(0,o) = o == 0 ? I : downscale(G(downscaleIndex,o-1), 2);
//#define METHOD_1
#ifdef METHOD_1
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int s = 1; s < numScales; ++s)
        G(s,o).resize(G(0,o).sizes());
#ifdef _OPENMP
# pragma omp parallel for
#endif
      for (int s = 1; s < numScales; ++s)
        applyGaussianFilter(G(s,o), G(0,o), initSigma*sqrt(pow(k,2*s) -1.f));
#else
      for (int s = 1; s < numScales; ++s)
      {
        Scalar sigma = sqrt(k*k*sigma_s_1*sigma_s_1 - sigma_s_1*sigma_s_1);
        G(s,o) = gaussian(G(s-1,o), sigma);
        sigma_s_1 *= k;
      }
#endif
    }

    // Done!
    return G;
  }
  //! Computes a pyramid of difference of Gaussians from the Gaussian pyramid.
  template <typename T>
  ImagePyramid<T> DoGPyramid(const ImagePyramid<T>& gaussians)
  {
    ImagePyramid<T> D;
    D.reset(gaussians.numOctaves(),
            gaussians.numScalesPerOctave()-1,
            gaussians.initScale(), 
            gaussians.scaleGeomFactor());

    for (int o = 0; o < D.numOctaves(); ++o)
    {
      D.octaveScalingFactor(o) = gaussians.octaveScalingFactor(o);
      for (int s = 0; s < D.numScalesPerOctave(); ++s)
      {
        D(s,o).resize(gaussians(s,o).sizes());
        D(s,o).array() = gaussians(s+1,o).array() - gaussians(s,o).array();
      }
    }
    return D;
  }
  //! Computes a pyramid of scale-normalized Laplacians of Gaussians from the 
  //! Gaussian pyramid.
  template <typename T>
  ImagePyramid<T> LoGPyramid(const ImagePyramid<T>& gaussians)
  {
    ImagePyramid<T> LoG;
    LoG.reset(gaussians.numOctaves(),
              gaussians.numScalesPerOctave(),
              gaussians.initScale(), 
              gaussians.scaleGeomFactor());

    for (int o = 0; o < LoG.numOctaves(); ++o)
    {
      LoG.octaveScalingFactor(o) = gaussians.octaveScalingFactor(o);
      for (int s = 0; s < LoG.numScalesPerOctave(); ++s)
      {
        LoG(s,o) = laplacian(gaussians(s,o));
        for (typename Image<T>::iterator it = LoG(s,o).begin();
             it != LoG(s,o).end(); ++it)
          *it *= pow(gaussians.octRelScale(s), 2);
      }
    }
    return LoG;
  }
  //! Computes the gradient vector of \f$I(x,y,\sigma)\f$ at \f$(x,y,\sigma)\f$, 
  //! where \f$\sigma = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales per 
  //! octave.
  template <typename T>
  Matrix<T, 3, 1> gradient(const ImagePyramid<T>& I, int x, int y, int s, int o)
  {
    // Sanity check
    if (x < 1 || x >= I(s,o).width()-1  ||
        y < 1 || y >= I(s,o).height()-1 ||
        s < 1 || s >= static_cast<int>(I(o).size())-1)
    {
      std::ostringstream msg;
      msg << "Fatal error: gradient out of range!" << std::endl;
      std::cerr << msg.str() << std::endl;
      throw std::out_of_range(msg.str());
    }

    Matrix<T, 3, 1> d;
    d(0) = (I(x+1,y  ,s  ,o) - I(x-1,y  ,s  ,o)) / T(2);
    d(1) = (I(x  ,y+1,s  ,o) - I(x  ,y-1,s  ,o)) / T(2);
    d(2) = (I(x  ,y  ,s+1,o) - I(x  ,y  ,s-1,o)) / T(2);
    return d;
  }
  //! Computes the hessian matrix of \f$I(x,y,\sigma)\f$ at \f$(x,y,\sigma)\f$,
  //! where \f$\sigma = 2^{s/S + o}\f$ where \f$S\f$ is the number of scales 
  //! per octave.
  template <typename T>
  Matrix<T, 3, 3> hessian(const ImagePyramid<T>& I, int x, int y, int s, int o)
  {
    // Sanity check
    if (x < 1 || x >= I(s,o).width()-1  ||
        y < 1 || y >= I(s,o).height()-1 ||
        s < 1 || s >= static_cast<int>(I(o).size())-1)
    {
      std::string msg("Fatal error: Hessian out of range!");
      std::cerr << msg << std::endl;
      throw std::out_of_range(msg);
    }

    Matrix<T, 3, 3> H;
    // Ixx
    H(0,0) = I(x+1,y,s,o) - T(2)*I(x,y,s,o) + I(x-1,y,s,o);
    // Iyy
    H(1,1) = I(x,y+1,s,o) - T(2)*I(x,y,s,o) + I(x,y-1,s,o);
    // Iss
    H(2,2) = I(x,y,s+1,o) - T(2)*I(x,y,s,o) + I(x,y,s-1,o);
    // Ixy = Iyx
    H(0,1) = H(1,0) = ( I(x+1,y+1,s,o) - I(x-1,y+1,s,o) 
                      - I(x+1,y-1,s,o) + I(x-1,y-1,s,o) ) / T(4);
    // Ixs = Isx
    H(0,2) = H(2,0) = ( I(x+1,y,s+1,o) - I(x-1,y,s+1,o)
                      - I(x+1,y,s-1,o) + I(x-1,y,s-1,o) ) / T(4);
    // Iys = Isy
    H(1,2) = H(2,1) = ( I(x,y+1,s+1,o) - I(x,y-1,s+1,o)
                      - I(x,y+1,s-1,o) + I(x,y-1,s-1,o) ) / T(4);
    // Done!
    return H;
  }

  //! @}

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_DOG_HPP */
