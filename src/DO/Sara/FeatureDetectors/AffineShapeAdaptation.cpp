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

#include <DO/Sara/FeatureDetectors.hpp>


using namespace std;


namespace DO { namespace Sara {

  AdaptFeatureAffinelyToLocalShape::
  AdaptFeatureAffinelyToLocalShape()
  {
    // Parameters
    patch_size_ = 19;
    gauss_trunc_factor_ = 3.f;
    affine_adaptation_max_iter_ = 10;
    // Debug only: view the magnified patch with the following zoom factor.
    _debug = false;
    _patch_zoom_factor = 2.f;
    // Memory allocation.
    _patch.resize(patch_size_, patch_size_);
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
  update_normalized_patch(const Image<float>& I,
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

    bool success = warp(_patch, I, A, 0.f, true);
    debug_display_normalized_patch(fact);

    return success;
  }

  Matrix2f
  AdaptFeatureAffinelyToLocalShape::
  compute_moment_matrix_from_patch()
  {
    Image<Vector2f> gradients;
    gradients = _patch.compute<Gradient>();
    debug_check_weighted_patch(gradients);
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
  compute_transform_from_moment_matrix(const Matrix2f& M,
  float& anisotropicRatio)
  {
    // Get the SVD decomposition of the second order moment matrix.
    JacobiSVD<Matrix2f> svd(M, ComputeFullU);
    Vector2f S{ svd.singularValues() };  // momentMatrix = U*S*V^T
    Matrix2f U{ svd.matrixU() };         // rotation matrix
    // Get the dilation factor for each axis.
    Vector2f radii{ S.cwiseSqrt().cwiseInverse() };
    Matrix2f T{ U*radii.asDiagonal() * U.transpose() };
    // Normalize w.r.t. to the largest axis radius.
    T *= 1.f/radii(1);
    // Store the anisotropic ratio.
    anisotropicRatio = radii(0)/radii(1);
    // Ok, done.
    return T;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  rescale_transform(Matrix2f& T)
  {
    JacobiSVD<Matrix2f> svd(T);
    Vector2f sv = svd.singularValues();
    T /= sv(0);
  }

  bool
  AdaptFeatureAffinelyToLocalShape::
  operator()(Matrix2f& affAdaptTransfmMat,
             const Image<float>& I,
             const OERegion& feature)
  {
    debug_create_window_to_view_patch();
    // The affine transform we want to estimate.
    Matrix2f U;
    U.setIdentity();
    // Iterative estimation from the image.
    for (int iter = 0; iter < affine_adaptation_max_iter_; ++iter)
    {
      debug_print_affine_adaptation_iteration(iter);
      // Get the normalized patch.
      if (!update_normalized_patch(I, feature, U))
      {
        debug_print_patch_touches_image_boundaries();
        debug_close_window_used_to_view_patch();
        return false;
      }
      // Estimate shape matrix.
      Matrix2f mu(compute_moment_matrix_from_patch());
      // Deduce the linear transform.
      float anisotropicRatio;
      Matrix2f delta_U(compute_transform_from_moment_matrix(mu, anisotropicRatio));
      // Accumulate the transform.
      U = delta_U*U;
      rescale_transform(U);
      debug_check_moment_matrix_and_transform(mu, delta_U, anisotropicRatio, U);
      // Instability check (cf. [Mikolajczyk & Schmid, ECCV 2002])
      if (1.f/anisotropicRatio > 6.f)
      {
        debug_close_window_used_to_view_patch();
        return false;
      }
      // Stopping criterion (cf. [Mikolajczyk & Schmid, ECCV 2002])
      if (1.f - anisotropicRatio < 0.05f)
        break;
    }
    debug_close_window_used_to_view_patch();

    // Return the shape matrix.
    affAdaptTransfmMat = U.inverse().transpose()*U.inverse();
    return true;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_create_window_to_view_patch()
  {
    // Open window to visualize the patch.
    if (_debug)
      set_active_window( create_window(
      _patch.width()*_patch_zoom_factor,
      _patch.height()*_patch_zoom_factor,
      "Image patch centered on the feature") );
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_print_affine_adaptation_iteration(int iter)
  {
    if (_debug)
      cout << endl << "Iteration " << iter << endl;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_display_normalized_patch(float fact)
  {
    if (_debug)
    {
      cout << "Factor = " << fact << endl;
      display(_patch, 0, 0, _patch_zoom_factor);
      get_key();
    }
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_check_weighted_patch(const Image<Vector2f>& gradients)
  {
    if (_debug)
    {
      // Check the weighted patch.
      Image<float> grad_magnitude( gradients.compute<SquaredNorm>() );
      Image<float> weighted_patch(gradients.sizes());
      weighted_patch.array() = grad_magnitude.array().sqrt()*gaussian_weight_.array();
      weighted_patch = color_rescale(weighted_patch);
      display(weighted_patch, 0, 0, _patch_zoom_factor);
      get_key();
    }
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_close_window_used_to_view_patch()
  {
    if (_debug)
      close_window();
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_check_moment_matrix_and_transform(const Matrix2f& mu,
                                          const Matrix2f& delta_U,
                                          float anisotropicRatio,
                                          const Matrix2f& U)
  {
    if (_debug)
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
  debug_print_patch_touches_image_boundaries()
  {
    if (_debug)
      cout << "The patch touches the image boundaries" << endl;
  }


} /* namespace Sara */
} /* namespace DO */
