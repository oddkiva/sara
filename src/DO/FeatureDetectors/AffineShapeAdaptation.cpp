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

#include <DO/FeatureDetectors.hpp>

using namespace std;

namespace DO {

  AdaptFeatureAffinelyToLocalShape::
  AdaptFeatureAffinelyToLocalShape()
  {
    // Parameters
    patch_size_ = 19;
    gauss_trunc_factor_ = 3.f;
    affine_adaptation_max_iter_ = 10;
    // Debug only: view the magnified patch with the following zoom factor.
    debug_ = false;
    patch_zoom_factor_ = 2.f;
    // Memory allocation.
    patch_.resize(patch_size_, patch_size_);
    gaussian_weight_.resize(patch_size_, patch_size_);
    // Precompute the Gaussian weight.
    float sigma = (0.5f*patch_size_) / gauss_trunc_factor_;
    float r = patch_size_/2.f;
    for (int y = 0; y < patch_size_; ++y)
    {
      for (int x = 0; x < patch_size_; ++x)
      {
        float u = x-r;
        float v = y-r;
        gaussian_weight_(x,y) = exp(-(u*u+v*v)/(2.f*sigma*sigma));
      }
    }
  }

  bool
  AdaptFeatureAffinelyToLocalShape::
  updateNormalizedPatch(const Image<float>& I,
                        const OERegion& feature,
                        const Matrix2f& T)
  {
    // The square image patch on which we estimate the second-moment matrix has
    // a side length $l$.
    // Let $(u,v)$ be a point in the domain $[0,l]^2$ of the image patch.

    // If $\mathbf{T}$ is the identity, we need to remap it to the original 
    // image domain $[x-3\sigma, x+3\sigma] \times [y-3\sigma, y+3\sigma]$,
    // where $(x,y,\sigma)$ denotes the feature location in scale-space.

    // So the transform is:
    // $\mathbf{A} (x,y) = \frac{3\sigma}{r} \mathbf{T} (x-r, y-r) + \mathbf{c}$
    // where:
    // - $\mathbf{c}$ is the center of the feature,
    // - $r = l/2$.
    Matrix3f A;
    float r = patch_size_/2.f;
    float fact = gauss_trunc_factor_*feature.scale()/r;

    A.block<2,2>(0,0) = T*fact;
    A.col(2) << T*Point2f(-r,-r)*fact+feature.center(), 1.f;
    A(2,0) = A(2,1) = 0.f;

    bool success = warp(patch_, I, A, 0.f, true);
    debug_displayNormalizedPatch(fact);

    return success;
  }

  Matrix2f
  AdaptFeatureAffinelyToLocalShape::
  computeMomentMatrixFromPatch()
  {
    Image<Vector2f> gradients;
    gradients = patch_.compute<Gradient>();
    debug_checkWeightedPatch(gradients);
    // Estimate the second moment matrix.
    Matrix2f moment;
    moment.setZero();
    for (int v = 0; v < patch_size_; ++v)
    {
      for (int u = 0; u < patch_size_; ++u)
      {
        float Ix = gradients(u,v)(0);
        float Iy = gradients(u,v)(1);
        moment(0,0) += gaussian_weight_(u,v)*Ix*Ix;
        moment(1,1) += gaussian_weight_(u,v)*Iy*Iy;
        moment(0,1) += gaussian_weight_(u,v)*Ix*Iy;
      }
    }
    moment(1,0) = moment(0,1);
    moment /= moment.norm();
    return moment;
  }

  Matrix2f
  AdaptFeatureAffinelyToLocalShape::
  computeTransformFromMomentMatrix(const Matrix2f& M,
                                   float& anisotropicRatio)
  {
    // Get the SVD decomposition of the second order moment matrix.
    JacobiSVD<Matrix2f> svd(M, ComputeFullU);
    Vector2f S = svd.singularValues();  // momentMatrix = U*S*V^T
    Matrix2f U = svd.matrixU();         // rotation matrix
    // Get the dilation factor for each axis.
    Vector2f radii( S.cwiseSqrt().cwiseInverse() );
    Matrix2f T( U*radii.asDiagonal()*U.transpose() );
    // Normalize w.r.t. to the largest axis radius.
    T *= 1.f/radii(1);
    // Store the anisotropic ratio.
    anisotropicRatio = radii(0)/radii(1);
    // Ok, done.
    return T;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  rescaleTransform(Matrix2f& T)
  {
    JacobiSVD<Matrix2f> svd(T);
    Vector2f sv = svd.singularValues();
    T /= sv(0);
  }

  bool
  AdaptFeatureAffinelyToLocalShape::
  operator()(Matrix2f& shapeMat,
             const Image<float>& I,
             const OERegion& feature)
  {
    debug_openWindowToViewPatch();
    // The affine transform we want to estimate.
    Matrix2f U;
    U.setIdentity();
    // Iterative estimation from the image.
    for (int iter = 0; iter < affine_adaptation_max_iter_; ++iter)
    {
      debug_printAffineAdaptationIteration(iter);
      // Get the normalized patch.
      if (!updateNormalizedPatch(I, feature, U))
      {
        debug_printPatchTouchesImageBoundaries();
        debug_closeWindowUsedToViewPatch();
        return false;
      }
      // Estimate shape matrix.
      Matrix2f mu(computeMomentMatrixFromPatch());
      // Deduce the linear transform.
      float anisotropicRatio;
      Matrix2f delta_U(computeTransformFromMomentMatrix(mu, anisotropicRatio));
      // Accumulate the transform.
      U = delta_U*U;
      rescaleTransform(U);
      debug_checkMomentMatrixAndTransform(mu, delta_U, anisotropicRatio, U);
      // Instability check (cf. [Mikolajczyk & Schmid, ECCV 2002])
      if (1.f/anisotropicRatio > 6.f)
      {
        debug_closeWindowUsedToViewPatch();
        return false;
      }
      // Stopping criterion (cf. [Mikolajczyk & Schmid, ECCV 2002])
      if (1.f - anisotropicRatio < 0.05f)
        break;
    }
    debug_closeWindowUsedToViewPatch();

    // Return the shape matrix.
    shapeMat = U.inverse().transpose()*U.inverse();
    return true;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_openWindowToViewPatch()
  {
    // Open window to visualize the patch.
    if (debug_)
      setActiveWindow( openWindow(
      patch_.width()*patch_zoom_factor_,
      patch_.height()*patch_zoom_factor_,
      "Image patch centered on the feature") );
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_printAffineAdaptationIteration( int iter )
  {
    if (debug_)
      cout << endl << "Iteration " << iter << endl;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_displayNormalizedPatch(float fact)
  {
    if (debug_)
    {
      cout << "Factor = " << fact << endl;
      display(patch_, 0, 0, patch_zoom_factor_);
      getKey();
    }
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_checkWeightedPatch(const Image<Vector2f>& gradients)
  {
    if (debug_)
    {
      // Check the weighted patch.
      Image<float> gradMag( gradients.compute<SquaredNorm>() );
      Image<float> weightedPatch(gradients.sizes());
      weightedPatch.array() = gradMag.array().sqrt()*gaussian_weight_.array();
      weightedPatch = colorRescale(weightedPatch);
      display(weightedPatch, 0, 0, patch_zoom_factor_);
      getKey();
    }
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_closeWindowUsedToViewPatch()
  {
    if (debug_)
      closeWindow();
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_checkMomentMatrixAndTransform(const Matrix2f& mu,
                                     const Matrix2f& delta_U,
                                     float anisotropicRatio,
                                     const Matrix2f& U)
  {
    if (debug_)
    {
      cout << "moment matrix = " << endl << mu << endl;
      cout << "delta_U = " << endl <<  delta_U << endl;
      if (1.f/anisotropicRatio > 6.f)
        cout << "WARNING: delta_U has excessive anisotropy!" << endl;
      cout << "U = " << endl <<  U << endl;
    }
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_printPatchTouchesImageBoundaries()
  {
    if (debug_)
      cout << "The patch touches the image boundaries" << endl;
  }

} /* namespace DO */