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

//! @file

#ifndef DO_SARA_FEATUREDETECTORS_AFFINEADAPTATION_HPP
#define DO_SARA_FEATUREDETECTORS_AFFINEADAPTATION_HPP

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/Image.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup FeatureDetectors
    @defgroup AffineAdaptation Affine Shape Adaptation
    @{
  */

  /*!
    @brief Functor that adapts the feature scale to the local shape from the
    second-moment matrix (cf. [Mikolajczyk & Schmid, ECCV 2002]).

    Given a feature \f$(\mathbf{x}, \sigma)\f$, the local affine shape adaptation
    is based on the estimation of the second moment matrix
    \f[
      \mu(\mathbf{x}, \sigma) =
      (g_\sigma) * \left[ (\nabla I) (\nabla I)^T \right] (\mathbf{x})
    \f]
    which is also the Mahalanobis distance reflecting the anisotropy of the
    local shape.
   */
  class DO_SARA_EXPORT AdaptFeatureAffinelyToLocalShape
  {
  public:
    /*!
      \todo: redo because there are hard-coded parameters in the default
     constructor.
     */
    AdaptFeatureAffinelyToLocalShape();

    /*!
      Estimates the local shape at some given point of an image.
      @param[in,out]
        affAdaptTransfmMat
        the affine adaptation transformation matrix, i.e., the Mahalanobis
        distance reflecting the anisotropy of the local shape
      @param[in] I the input image
      @param[in]
        feature
        the point on which we estimate the local shape in image I.
     */
    bool operator()(Matrix2f& affine_adaptation_transform,
                    const Image<float>& image,
                    const OERegion& feature);

  private:
    /*!
      Return:
      - true if the normalized patch does not touch the image boundaries.
      - false otherwise.
      If false is returned, then
     */
    bool warp_patch(const Image<float>& src,
                    Image<float>& dst,
                    const Matrix3f& homography_from_dst_to_src);

    /*!
      Return:
      - true if the normalized patch does not touch the image boundaries.
      - false otherwise.
      If false is returned, then
     */
    bool update_normalized_patch(const Image<float>& image,
                                 const OERegion& feature,
                                 const Matrix2f& affine_adaptation_transform);

    /*!
      Given a feature $(\mathbf{x}, \sigma)\f$, computes \f$\mathbf{x}\f$ at
      the second moment matrix \f$mu(x, \sigma)\f$ defined as
      \f[
        mu(\mathbf{x}, \sigma) =
        (g_\sigma) * (\nabla I) (\nabla I)^T (\mathbf{x})
      \f]
     */
    Matrix2f compute_moment_matrix_from_patch();

    //! Find one linear transform associated to the second moment matrix.
    Matrix2f compute_transform_from_moment_matrix(const Matrix2f& moment_matrix,
                                                  float& anisotropic_ratio);

    //! Normalize the transform
    void rescale_transform(Matrix2f& transform);

  private: /* debugging methods. */
    void debug_create_window_to_view_patch();
    void debug_print_affine_adaptation_iteration(int iter);
    void debug_print_patch_touches_image_boundaries();
    void debug_display_normalized_patch(float scale);
    void debug_check_weighted_patch(const Image<Vector2f>& gradients);
    void debug_check_moment_matrix_and_transform(const Matrix2f& mu,
                                                 const Matrix2f& delta_U,
                                                 float anisotropic_ratio,
                                                 const Matrix2f& U);
    void debug_close_window_used_to_view_patch();

  private:
    int _patch_size;
    float _gauss_trunc_factor;
    int affine_adaptation_max_iter_;
    Image<float> _patch;
    Image<float> _gaussian_weights;

    float _patch_zoom_factor;
    bool _debug;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDETECTORS_AFFINEADAPTATION_HPP */