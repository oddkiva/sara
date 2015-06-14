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


namespace DO { namespace Sara {

  /*!
    \ingroup FeatureDetectors
    \defgroup AffineAdaptation Affine Shape Adaptation
    @{
  */

  /*!
    \brief Functor that adapts the feature scale to the local shape from the
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
  class AdaptFeatureAffinelyToLocalShape
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
    bool operator()(Matrix2f& affAdaptTransfmMat,
                    const Image<float>& I,
                    const OERegion& feature);
  private:
    /*!
      Returns:
      - true if the normalized patch does not touch the image boundaries.
      - false otherwise.
      If false is returned, then
     */
    bool updateNormalizedPatch(const Image<float>& I,
                               const OERegion& feature,
                               const Matrix2f& T);
    /*!
      Given a feature $(\mathbf{x}, \sigma)\f$, computes \f$\mathbf{x}\f$ at
      the second moment matrix \f$mu(x, \sigma)\f$ defined as
      \f[
        mu(\mathbf{x}, \sigma) =
        (g_\sigma) * (\nabla I) (\nabla I)^T (\mathbf{x})
      \f]
     */
    Matrix2f computeMomentMatrixFromPatch();
    //! Find one linear transform associated to the second moment matrix.
    Matrix2f computeTransformFromMomentMatrix(const Matrix2f& momentMatrix,
                                              float& anisotropicRatio);
    //! Normalize the transform
    void rescaleTransform(Matrix2f& T);
  private: /* debugging methods. */
    void debug_openWindowToViewPatch();
    void debug_printAffineAdaptationIteration(int iter);
    void debug_printPatchTouchesImageBoundaries();
    void debug_displayNormalizedPatch(float fact);
    void debug_checkWeightedPatch(const Image<Vector2f>& gradients);
    void debug_checkMomentMatrixAndTransform(const Matrix2f& mu,
                                            const Matrix2f& delta_U,
                                            float anisotropicRatio,
                                            const Matrix2f& U);
    void debug_closeWindowUsedToViewPatch();
  private:
    int patch_size_;
    float gauss_trunc_factor_;
    int affine_adaptation_max_iter_;
    Image<float> patch_;
    Image<float> gaussian_weight_;

    float patch_zoom_factor_;
    bool debug_;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATUREDETECTORS_AFFINEADAPTATION_HPP */
