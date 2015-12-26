// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors.hpp>

#include <DO/Sara/Geometry/Tools/Utilities.hpp>


using namespace std;


namespace DO { namespace Sara {

  AdaptFeatureAffinelyToLocalShape::
  AdaptFeatureAffinelyToLocalShape()
  {
    // Parameters
    _patch_size = 19;
    _gauss_trunc_factor = 3.f;
    affine_adaptation_max_iter_ = 10;

    // Debug only: view the magnified patch with the following zoom factor.
    _debug = false;
    _patch_zoom_factor = 2.f;

    // Memory allocation.
    _patch.resize(_patch_size, _patch_size);
    _gaussian_weights.resize(_patch_size, _patch_size);

    // Precompute the Gaussian weight.
    auto sigma = (0.5f*_patch_size) / _gauss_trunc_factor;
    auto r = _patch_size / 2.f;
    for (int y = 0; y < _patch_size; ++y)
    {
      for (int x = 0; x < _patch_size; ++x)
      {
        const auto u = x - r;
        const auto v = y - r;
        _gaussian_weights(x, y) = exp(-(u*u + v*v) / (2.f*sigma*sigma));
      }
    }
  }

  bool
  AdaptFeatureAffinelyToLocalShape::warp_patch(
    const Image<float>& image,
    Image<float>& patch,
    const Matrix3f& homography_from_dst_to_src)
  {
    const Matrix3f& H = homography_from_dst_to_src;

    for (auto it = patch.begin_array(); !it.end(); ++it)
    {
      // Get the corresponding coordinates in the source image.
      Vector3f H_P;
      H_P = H * (Vector3f() << it.position().cast<float>(), 1).finished();
      H_P /= H_P(2);

      // Check if the position is not in the src domain [0,w[ x [0,h[.
      bool position_is_in_image_domain =
        H_P.x() >= 0 && H_P.x() < float(image.width()) &&
        H_P.y() >= 0 && H_P.y() < float(image.height());

      // Fill with either the default value or the interpolated value.
      if (position_is_in_image_domain)
      {
        Vector2d H_p(H_P.template head<2>().template cast<double>());
        auto pixel_value = interpolate(image, H_p);
        *it = float(pixel_value);
      }
      else
        return false;
    }

    return true;
  }

  bool
  AdaptFeatureAffinelyToLocalShape::
  update_normalized_patch(const Image<float>& I,
                          const OERegion& f,
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
    auto r = _patch_size / 2.f;
    auto s = _gauss_trunc_factor*f.scale() / r;

    A.block<2, 2>(0, 0) = T*s;
    A.col(2) << T*Point2f(-r, -r)*s + f.center(), 1.f;
    A.row(2).head(2).fill(0.f);

    auto success = warp_patch(I, _patch, A);
    debug_display_normalized_patch(s);

    return success;
  }

  Matrix2f
  AdaptFeatureAffinelyToLocalShape::
  compute_moment_matrix_from_patch()
  {
    auto gradients = _patch.compute<Gradient>();
    debug_check_weighted_patch(gradients);
    // Estimate the second moment matrix.
    Matrix2f moment;
    moment.setZero();
    for (int v = 0; v < _patch_size; ++v)
    {
      for (int u = 0; u < _patch_size; ++u)
      {
        float Ix = gradients(u,v)(0);
        float Iy = gradients(u,v)(1);
        moment(0,0) += _gaussian_weights(u,v)*Ix*Ix;
        moment(1,1) += _gaussian_weights(u,v)*Iy*Iy;
        moment(0,1) += _gaussian_weights(u,v)*Ix*Iy;
      }
    }
    moment(1,0) = moment(0,1);
    moment /= moment.norm();
    return moment;
  }

  Matrix2f
  AdaptFeatureAffinelyToLocalShape::
  compute_transform_from_moment_matrix(const Matrix2f& M,
                                       float& anisotropic_ratio)
  {
    // Get the SVD decomposition of the second order moment matrix.
    JacobiSVD<Matrix2f> svd(M, ComputeFullU);
    Vector2f S{ svd.singularValues() };  // moment matrix = U*S*V^T
    Matrix2f U{ svd.matrixU() };         // rotation matrix

    // Get the dilation factor for each axis.
    Vector2f radii{ S.cwiseSqrt().cwiseInverse() };
    Matrix2f T{ U*radii.asDiagonal() * U.transpose() };

    // Normalize w.r.t. to the largest axis radius.
    T *= 1.f / radii(1);

    // Store the anisotropic ratio.
    anisotropic_ratio = radii(0) / radii(1);

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
  operator()(Matrix2f& affine_adapt_transform,
             const Image<float>& image,
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
      if (!update_normalized_patch(image, feature, U))
      {
        debug_print_patch_touches_image_boundaries();
        debug_close_window_used_to_view_patch();
        return false;
      }

      // Estimate shape matrix.
      Matrix2f mu(compute_moment_matrix_from_patch());

      // Deduce the linear transform.
      float anisotropic_ratio;
      Matrix2f delta_U(compute_transform_from_moment_matrix(mu, anisotropic_ratio));

      // Accumulate the transform.
      U = delta_U*U;
      rescale_transform(U);
      debug_check_moment_matrix_and_transform(mu, delta_U, anisotropic_ratio, U);

      // Instability check (cf. [Mikolajczyk & Schmid, ECCV 2002])
      if (1.f / anisotropic_ratio > 6.f)
      {
        debug_close_window_used_to_view_patch();
        return false;
      }

      // Stopping criterion (cf. [Mikolajczyk & Schmid, ECCV 2002])
      if (1.f - anisotropic_ratio < 0.05f)
        break;
    }

    debug_close_window_used_to_view_patch();

    // Return the shape matrix.
    affine_adapt_transform = U.inverse().transpose()*U.inverse();
    return true;
  }

  void
  AdaptFeatureAffinelyToLocalShape::
  debug_create_window_to_view_patch()
  {
    // Open window to visualize the patch.
    if (_debug)
    {
      auto window = create_window(
        int_round(_patch.width()*_patch_zoom_factor),
        int_round(_patch.height()*_patch_zoom_factor),
        "Image patch centered on the feature");
      set_active_window(window);
    }
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
      auto grad_magnitude = gradients.compute<SquaredNorm>();
      auto weighted_patch = Image<float>{ gradients.sizes() };
      weighted_patch.array() = grad_magnitude.array().sqrt()*_gaussian_weights.array();
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
                                          float anisotropic_ratio,
                                          const Matrix2f& U)
  {
    if (_debug)
    {
      cout << "moment matrix = " << endl << mu << endl;
      cout << "delta_U = " << endl << delta_U << endl;
      if (1.f / anisotropic_ratio > 6.f)
        cout << "WARNING: delta_U has excessive anisotropy!" << endl;
      cout << "U = " << endl << U << endl;
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