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

#ifndef DO_FEATUREDESCRIPTORS_SIFT_HPP
#define DO_FEATUREDESCRIPTORS_SIFT_HPP

namespace DO {

  /*!
    \ingroup FeatureDescriptors
    \defgroup Descriptors
    @{
   */

  //! \brief Functor class used to compute the SIFT Descriptor at some location.
  template <int N=4, int O=8>
  class ComputeSIFTDescriptor
  {
  public: /* interface. */
    enum { Dim = N*N*O };
    typedef Matrix<float, Dim, 1> SIFT;
    //! Constructor.
    ComputeSIFTDescriptor(float binScaleUnitLength = 3.f,
                          float maxBinValue = 0.2f)
      : bin_scale_unit_length_(binScaleUnitLength), max_bin_value_(maxBinValue) {}
    //! Computes the SIFT descriptor for keypoint \$(x,y,\sigma,\theta)\f$.
    SIFT operator()(float x, float y, float sigma, float theta,
                    const Image<Vector2f>& gradPolar) const
    {
      const float pi = static_cast<float>(M_PI);
      /*
        The oriented keypoint is denoted by $k = (x,y,\sigma,\theta)$.
        SIFT describes keypoint $k$ in a similarity-invariant manner.

        To do so, we consider a square image patch which:
        - is centered in $(x,y)$
        - has an orientation angle $\theta$ w.r.t. the image frame coordinates:
        => to ensure rotation invariance
        - has a side length proportional to the scale $\sigma$:
        => to ensure scale invariance
        This square patch is denoted by $P(x,y,\sigma,\theta) = P(k)$.

        The square patch $P(x,y,\sigma,\theta)$ is itself divided into NxN
        smaller square patches $(P_{i,j})_{1 \leq i \leq N, j \leq j \leq N}$.

        Notice that we omit the variables $(x,y,\sigma,\theta)$ which the
        patches $P_{i,j}$ actually depend on.

        $N$ corresponds to the template argument 'int N' which should be 4 as 
        stated in the paper [Lowe, IJCV 2004]).

        In the image, each small square patch $P_{i,j}$ has a side length $l$ 
        proportional to the scale $\sigma$ of the keypoint, i.e., 
        $l = \lambda \sigma$.
      */
      const float lambda = bin_scale_unit_length_;
      const float l = lambda*sigma;
      /*
        It is important to note that $\lambda$ is some 'universal' constant 
        used for all SIFT descriptors to ensure the scale-invariance of the 
        descriptor.
      */

      /*
        Now in each image square patch $P_{i,j}$, we build a histogram of 
        gradient orientations $\mathbf{h}_{i,j} \in \mathbb{R}^d$, which 
        quantizes the gradient orientations into $O$ principal orientations.
        $O$ corresponds to the template argument 'int O'.

        Let us initialize the SIFT descriptor consisting of the NxN histograms 
        $\mathbf{h}_{i,j}$, each in $\mathbf{R}^O$ as follows.
      */
      SIFT h(SIFT::Zero());

      /*
       In the rescaled and oriented coordinate frame bound to the patch $P(k)$, 
       - keypoint $k$ is located at (0,0)
       - centers $C_{i,j}$ of patch $P_{i,j}$ are located at
         $[ -(N+1)/2 + i, -(N+1)/2 + j ]$
      
         For example for $N=4$, they are at:
         (-1.5,-1.5) (-0.5,-1.5) (0.5,-1.5) (1.5,-1.5)
         (-1.5,-0.5) (-0.5,-0.5) (0.5,-0.5) (1.5,-0.5)
         (-1.5, 0.5) (-0.5, 0.5) (0.5, 0.5) (1.5, 0.5)
         (-1.5, 1.5) (-0.5, 1.5) (0.5, 1.5) (1.5, 1.5)
      
       Gradients in $[x_i-1, x_i+1] \times [y_i-1, y_i+1]$ contributes
       to histogram $\mathbf{h}_{i,j}$, namely gradients in the square patch
       $Q_{i,j}$
       - centered in $C_{i,j}$ as square patch $P_{i,j}$,
       - with side length $2$.
       That is because we want to do trilinear interpolation in order to make 
       SIFT robust to small shift in rotation, translation.

       Therefore, to compute the SIFT descriptor we need to scan all the pixels 
       on a larger circular image patch with radius $r$:
      */
      const float r = sqrt(2.f) * l * (N+1)/2.f;
      /*
       In the above formula, notice:
       - the factor $\sqrt{2}$ because diagonal corners of the furthest patches 
         $P_{i,j}$ from the center $(x,y)$ must be in the circular patch.
       - the factor $(N+1)/2$ because we have to include the gradients in larger
         patches $Q_{i,j}$ for each $P_{i,j}$.
       It is recommended to make a drawing to convince oneself.
      */

      // To build the SIFT descriptor, we do the following procedure:
      // - we work in the image reference frame;
      // - we scan in the convolved image $G_\sigma$ the position $(x+u, y+v)$
      //   where $(u,v) \in [-r,r]^2$;
      // - we retrieve its coordinates in the oriented frame of the patch 
      //   $P(x,y,\sigma,\theta)$ with inverse transform $T = 1/l R_\theta^T$
      Matrix2f T;
      T << cos(theta), sin(theta),
          -sin(theta), cos(theta);
      T /= l;
      // Loop to perform interpolation
      const int rounded_r = intRound(r);
      const float rounded_x = intRound(x);
      const float rounded_y = intRound(y);
      for (int v = -rounded_r; v <= rounded_r; ++v)
      {
        for (int u = -rounded_r; u <= rounded_r; ++u)
        {
          // Compute the coordinates in the rescaled and oriented coordinate 
          // frame bound to patch $P(k)$.
          Vector2f pos( T*Vector2f(u,v) );
          // subpixel correction?
          /*pos.x() -= (x - rounded_x);
          pos.y() -= (y - rounded_y);*/

          if ( rounded_x+u < 0 || rounded_x+u >= gradPolar.width()  ||
               rounded_y+v < 0 || rounded_y+v >= gradPolar.height() )
            continue;

          // Compute the Gaussian weight which gives more emphasis to gradient 
          // closer to the center.
          float weight = exp(-pos.squaredNorm()/(2.f*pow(N/2.f, 2)));
          float mag = gradPolar(rounded_x+u, rounded_y+v)(0);
          float ori = gradPolar(rounded_x+u, rounded_y+v)(1) - theta;
          ori = ori < 0.f ? ori+2.f*pi : ori;
          ori *= float(O)/(2.f*pi);

          // The coordinate frame is centered in the patch center, thus:
          // $(x,y)$ is in $[-(N+1)/2, (N+1)/2]^2$.
          //
          // Change the coordinate frame so that $(x,y)$ is in $[-1, N]^2$. Thus,
          // translate by $[ (N-1)/2, (N-1)/2 ]$.
          pos.array() += N/2.f - 0.5f;
          if (pos.minCoeff() <= -1.f  || pos.maxCoeff() >= static_cast<float>(N))
            continue;
          // In the translated coordinate frame, note that for $N=4$ the centers
          // are now located at:
          //   (0,0) (1,0) (2,0) (3,0)
          //   (0,1) (1,1) (2,1) (3,1)
          //   (0,2) (1,1) (2,2) (3,2)
          //   (0,3) (1,1) (2,3) (3,3)
          //

          // Update the SIFT descriptor using trilinear interpolation.
          accumulate(h, pos, ori, weight, mag);
        }
      }
    
      h.normalize();

      h = (h * 512.f).cwiseMin(Matrix<float, Dim, 1>::Ones()*255.f);
      return h;
    }
    //! Helper member function.
    SIFT operator()(const OERegion& f, const Image<Vector2f>& gradPolar) const
    { return this->operator()(f.x(), f.y(), f.scale(), f.orientation(), gradPolar); }
    //! Helper member function.
    std::vector<SIFT> operator()(const std::vector<OERegion>& features,
                                 const std::vector<Point2i>& scaleOctavePairs,
                                 const ImagePyramid<Vector2f>& gradPolars) const
    {
      std::vector<SIFT> sifts;
      sifts.resize(features.size());
      for (size_t i = 0; i != features.size(); ++i)
      {
        sifts[i] = this->operator()(
          features[i],
          gradPolars(scaleOctavePairs[i](0), scaleOctavePairs[i](1)) );
      }
      return sifts;
    }
  public: /* debugging functions. */
    //! Check the grid on which we are drawing.
    void drawGrid(float x, float y, float sigma, float theta, float octScaleFactor,
                   int penWidth = 1)
    {
      const float lambda = 3.f;
      const float l = lambda*sigma;
      Vector2f grid[N+1][N+1];
      Matrix2f T;
      theta = 0;
      T << cos(theta),-sin(theta),
           sin(theta), cos(theta);
      T *= l;
      for (int v = 0; v < N+1; ++v)
        for (int u = 0; u < N+1; ++u)
          grid[u][v] = (Vector2f(x,y) + T*Vector2f(u-N/2.f,v-N/2.f))*octScaleFactor;
      for (int i = 0; i < N+1; ++i)
        drawLine(grid[0][i], grid[N][i], Green8, penWidth);
      for (int i = 0; i < N+1; ++i)
        drawLine(grid[i][0], grid[i][N], Green8, penWidth);
    
      Vector2f a(x,y);
      a *= octScaleFactor;
      Vector2f b;
      b = a+octScaleFactor*N/2.f*T*Vector2f(1,0);
      drawLine(a, b, Red8, penWidth+2);
    }
  private: /* member functions. */
    //! The accumulation function based on trilinear interpolation.
    void accumulate(SIFT& h, const Vector2f& pos, float ori,
                    float weight, float mag) const
    {
      // By trilinear interpolation, we mean that in this translated coordinate
      // frame, a gradient with orientation $\theta$ and located at 
      // $(x,y) \in [-1,N]^2$ contributes to the 4 histograms:
      //  - $\mathbf{h}_{ floor(y)  , floor(x)  }$
      //  - $\mathbf{h}_{ floor(y)  , floor(x)+1}$
      //  - $\mathbf{h}_{ floor(y)+1, floor(x)  }$
      //  - $\mathbf{h}_{ floor(y)+1, floor(x)+1}$
      // In each of these histograms, the following bins are accumulated:
      //  - $\mathbf{h}_{o}$
      //  - $\mathbf{h}_{o+1}$
      // where $o = floor(\theta * O/ (2*\pi))$
      //
      // Note that a gradient at the boundary like $(-1,-1)$ contributes only
      // to P_{0,0}.
      float xfrac = pos.x() - floor(pos.x());
      float yfrac = pos.y() - floor(pos.y());
      float orifrac = ori - floor(ori);
      int xi = int(pos.x());
      int yi = int(pos.y());
      int orii = int(ori);
      for (int dy = 0; dy < 2; ++dy)
      {
        int y = yi+dy;
        if (y < 0 || y >= N)
          continue;
        float wy = (dy == 0) ? 1.f-yfrac : yfrac;
        for (int dx = 0; dx < 2; ++dx)
        {
          int x = xi+dx;
          if (x < 0 || x >= N)
            continue;
          float wx = (dx == 0) ? 1.f-xfrac : xfrac;
          for (int dori = 0; dori < 2; ++dori)
          {
            int o = (orii+dori)%O;
            float wo = (dori == 0) ? 1.f-orifrac : orifrac;
            // Trilinear interpolation: 
            // SIFT(y,x,o) += wy*wx*wo*weight*mag;
            h[at(y,x,o)] += wy*wx*wo*weight*mag;
          }
        }
      }
    }
    //! Normalize in a contrast-invariant way.
    void normalize(SIFT& h)
    {
      // Euclidean normalization.
      h.normalize();
      // Clamp histogram bin values $h_i$ to 0.2 for enhanced robustness to 
      // lighting change.
      h = h.cwiseMin(SIFT::Ones()*max_bin_value_);
      // Renormalize again.
      h.normalize();
    }
    //! Helper access function.
    inline int at(int i, int j, int o) const
    { return N*O*i + j*O + o; }
  private: /* data members. */
    float bin_scale_unit_length_;
    float max_bin_value_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDESCRIPTORS_SIFT_HPP */
