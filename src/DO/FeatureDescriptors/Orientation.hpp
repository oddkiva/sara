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

#ifndef DO_FEATUREDESCRIPTORS_ORIENTATION_HPP
#define DO_FEATUREDESCRIPTORS_ORIENTATION_HPP

namespace DO {

  /*!
    \ingroup FeatureDesriptors
    \defgroup Orientation Dominant Orientation Assignment
    @{
  */

  /*!
    \brief Computes the image gradients of image \f$I\f$ in polar coordinates.
    
    In each pixel \f$(x,y) \in [0,w[ \times [0,h[\f$, 
    \f$\nabla I(x,y)\f$ is the 2D vector \f$(r,\theta)\f$ where:
    - \f$r = 2 ||\nabla I(x,y)||\f$,
    - \f$\theta = \mathrm{angle}( \nabla I(x,y) )\f$.
   */
  template <typename T>
  Image<Matrix<T,2,1> > gradPolar(const Image<T>& I)
  {
    Image<Matrix<T,2,1> > g;
    gradient(g, I);
    for (typename Image<Matrix<T,2,1> >::iterator it = g.begin(); 
         it != g.end(); ++it)
    {
      float r = 2*it->norm();
      float theta = atan2(it->y(), it->x());
      *it = Vector2f(r, theta);
    }
    return g;
  }
  /*!
    \brief Computes the image gradients in polar coordinates for each image in 
    the pyramid.
   */
  template <typename T>
  ImagePyramid<Matrix<T, 2, 1> > gradPolar(const ImagePyramid<T>& pyramid)
  {
    ImagePyramid<Matrix<T, 2, 1> > pyramidOfGradients;
    pyramidOfGradients.reset(
      pyramid.numOctaves(),
      pyramid.numScalesPerOctave(),
      pyramid.initScale(),
      pyramid.scaleGeomFactor() );
    for (int o = 0; o < pyramid.numOctaves(); ++o)
    {
      pyramidOfGradients.octaveScalingFactor(o) = pyramid.octaveScalingFactor(o);
      for (int s = 0; s < pyramid.numScalesPerOctave(); ++s)
        pyramidOfGradients(s,o) = gradPolar(pyramid(s,o));
    }
    return pyramidOfGradients;
  }
  /*!
    \brief  Computes the orientation histogram on a local patch around keypoint 
    \f$(x,y,\sigma)\f$.
   */
  template <typename T, int N>
  void computeOrientationHistogram(Array<T, N, 1>& oriHist,
                                   const Image<Matrix<T,2,1> >& gradPolar,
                                   T x, T y, T s,
                                   int patchTruncationFactor = 3,
                                   T blurFactor = T(1.5))
  {
    // Weighted histogram of gradients.
    oriHist.setZero();

    // Rounding of the coordinates.
    int xi = intRound(x);
    int yi = intRound(y);

    // std deviation of the gaussian weight (cf. [Lowe, IJCV 2004])
    T sigma = s*blurFactor;

    // Patch radius on which the histogram of gradients is performed.
    int patchRadius = intRound(sigma*patchTruncationFactor);

    // Accumulate the histogram of orientations.
    for (int v = -patchRadius; v <= patchRadius; ++v)
    {
      for (int u = -patchRadius; u <= patchRadius; ++u)
      {
        if ( xi+u < 0 || xi+u >= gradPolar.width()  ||
             yi+v < 0 || yi+v >= gradPolar.height() )
          continue;

        T mag = gradPolar(xi+u, yi+v)(0);
        T ori = gradPolar(xi+u, yi+v)(1);
        // ori is in \f$]-\pi, \pi]\f$, so translate ori by \f$2*\pi\f$ if it is 
        // negative.
#ifndef LOWE
        ori = ori < 0 ? ori+T(2.*M_PI) : ori;
        int binIndex = floor(ori/T(2*M_PI) * T(N));
        binIndex %= N;
#else
        int binIndex = int( (N * (ori + M_PI + 0.001f) / (2.0f * M_PI)) );
        binIndex = std::min(binIndex, N - 1);
#endif
        if (binIndex < 0 || binIndex >= N)
        {
          std::ostringstream oss;
          oss << "Orientation bin index out of range: " << binIndex 
              << " theta = " << ori << std::endl;
          std::cerr << oss.str() << std::endl;
          throw std::out_of_range(oss.str());
        }


        // Give more emphasis to gradient orientations that lie closer to the
        // keypoint location.
        T weight = exp(-(u*u+v*v)/(T(2)*sigma*sigma));
        // Also give more emphasis to gradient with large magnitude.
        oriHist( binIndex ) += weight*mag;
      }
    }
  }
  /*!
    \brief This is used in [Lowe, IJCV 2004] to determine keypoint orientations.
    
    Basically, the histogram is convolved 6 times with the mean kernel 
    \f$[1/3, 1/3, 1/3]\f$.
   */
  template <typename T, int N>
  void smoothHistogram_Lowe(Array<T, N, 1>& oriHist)
  {
    for (int iter = 0; iter < 6; ++iter)
    {
      float first = oriHist(0);
      float prev = oriHist(N-1);
      for (int i = 0; i < N-1; ++i)
      {
        float val = (prev+oriHist(i)+oriHist(i+1))/3.f;
        prev = oriHist(i);
        oriHist(i) = val;
      }
      oriHist(N-1) = (prev+oriHist(N-1)+first)/3.f;
    }
  }

  /*!
    \brief  This is used in [Lowe, IJCV 2004] to find histogram peaks.
    
    A histogram peak is by definition the index \f$i\f$ such that:
    - \f$h_{i} > h_{i+1}\f$
    - \f$h_{i} > h_{i-1}\f$
    
    Only histogram peaks \f$i\f$ such that \f$h_i \geq 0.8 \max_j h_j\f$
   */
  template <typename T, int N>
  std::vector<int> findPeaks(const Array<T, N, 1>& oriHist,
                             T peakRatioThres = T(0.8))
  {
    T max = oriHist.maxCoeff();
    std::vector<int> oriPeaks;
    oriPeaks.reserve(N);
    for (int i = 0; i < N; ++i)
      if ( oriHist(i) >= peakRatioThres*max &&
           oriHist(i) > oriHist((i-1+N)%N)     &&
           oriHist(i) > oriHist((i+1)%N)       )
        oriPeaks.push_back(i);
    return oriPeaks;
  }
  /*!
    \brief Refine peaks as in [Lowe, IJCV 2004] by interpolation based on a 
    second-order Taylor approximation.
   */
  template <typename T, int N>
  T refinePeak(const Array<T, N, 1>& oriHist, int i)
  {
    T y0 = oriHist( (i-1+N)%N );
    T y1 = oriHist( i );
    T y2 = oriHist( (i+1)%N );
    // Denote the orientation histogram function by \f$f\f$.
    // perform a 2nd-order Taylor approximation:
    // \f$f(x+h) = f(x) + f'(x)h + f''(x) h^2/2\f$
    // We approximate \f$f'\f$ and \f$f''\f$ by finite difference.
    T fprime = (y2-y0) / 2.f;
    T fsecond = y0 - 2.f*y1 + y2;
    // Maximize w.r.t. to \f$h\f$, derive the expression.
    // Thus \f$h = -f'(x)/f''(x)\f$.
    T h = -fprime / fsecond;
    // Add the offset \f$h\f$ to get the refined orientation value.
    // Note that we also add the 0.5f offset, because samples are assumed taken 
    // on the middle of the interval \f$[i, i+1)\f$.
    return T(i)+T(0.5)+h;
  }
  //! \brief Helper functions.
  template <typename T, int N>
  std::vector<T> refinePeaks(const Array<T, N, 1>& oriHist,
                             const std::vector<int>& ori)
  {
    std::vector<T> oriT(ori.size());
    for (size_t i = 0; i < ori.size(); ++i)
      oriT[i] = refinePeak<T,N>(oriHist, ori[i]);
    return oriT;
  }

  //! \brief Basically a helper class.
  class ComputeDominantOrientations
  {
  public:
    ComputeDominantOrientations(float peakRatioThres = 0.8f,
                                float patchTruncationFactor = 3.f,
                                float blurFactor = 1.5f);

    std::vector<float> operator()(const Image<Vector2f>& gradients,
                                  float x, float y, float sigma) const;
  
    std::vector<float> operator()(const ImagePyramid<Vector2f>& pyramid,
                                  const OERegion& extremum,
                                  const Point2i& scaleOctPair) const;

    void operator()(const ImagePyramid<Vector2f>& pyramid,
                    std::vector<OERegion>& extrema,
                    std::vector<Point2i>& scaleOctPairs) const;

  private:
    float peak_ratio_thres_;
    float patch_truncation_factor_;
    float blur_factor_;
  };

  //! @}


} /* namespace DO */

#endif /* DO_FEATUREDESCRIPTORS_ORIENTATION_HPP */
