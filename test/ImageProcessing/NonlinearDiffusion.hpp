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

//=============================================================================
//
// NonlinearDiffusion.hpp
// Authors: Pablo F. Alcantarilla (1), Jesus Nuevo (2)
// Institutions: Georgia Institute of Technology (1)
//               TrueVision Solutions (2)
// Date: 15/09/2013
// Email: pablofdezalc@gmail.com
//
// AKAZE Features Copyright 2013, Pablo F. Alcantarilla, Jesus Nuevo
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file NonlinearDiffusion.hpp
 * @brief Functions for nonlinear diffusion filtering applications
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#ifndef AKAZE_NONLINEAR_DIFFUSION_HPP
#define AKAZE_NONLINEAR_DIFFUSION_HPP

#include <DO/Core.hpp>

namespace DO { namespace AKAZE {

/**
 * @brief This function smoothes an image with a Gaussian kernel
 * @param src Input image
 * @param dst Output image
 * @param ksize_x Kernel size in X-direction (horizontal)
 * @param ksize_y Kernel size in Y-direction (vertical)
 * @param sigma Kernel standard deviation
 */
void Gaussian_2D_Convolution(const Image<float>& src, Image<float>& dst,
                             size_t ksize_x, size_t ksize_y,
                             const float sigma);
/**
 * @brief This function computes image derivatives with Scharr kernel
 * @param src Input image
 * @param dst Output image
 * @param xorder Derivative order in X-direction (horizontal)
 * @param yorder Derivative order in Y-direction (vertical)
 * @note Scharr operator approximates better rotation invariance than
 * other stencils such as Sobel. See Weickert and Scharr,
 * A Scheme for Coherence-Enhancing Diffusion Filtering with Optimized Rotation Invariance,
 * Journal of Visual Communication and Image Representation 2002
 */
void Image_Derivatives_Scharr(const Image<float>& src, Image<float>& dst,
                              size_t xorder, size_t yorder);
/**
 * @brief This function computes the Perona and Malik conductivity coefficient g1
 * g1 = exp(-|dL|^2/k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void PM_G1(const Image<float>& Lx, const Image<float>& Ly, Image<float>& dst,
           const float k);
/**
 * @brief This function computes the Perona and Malik conductivity coefficient g2
 * g2 = 1 / (1 + dL^2 / k^2)
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 */
void PM_G2(const Image<float>& Lx, const Image<float>& Ly, Image<float>& dst, const float k);
/**
 * @brief This function computes Weickert conductivity coefficient g3
 * @param Lx First order image derivative in X-direction (horizontal)
 * @param Ly First order image derivative in Y-direction (vertical)
 * @param dst Output image
 * @param k Contrast factor parameter
 * @note For more information check the following paper: J. Weickert
 * Applications of nonlinear diffusion in image processing and computer vision,
 * Proceedings of Algorithmy 2000
 */
void Weickert_Diffusivity(const Image<float>& Lx, const Image<float>& Ly, Image<float>& dst,
                          const float k);
///**
// * @brief This function computes a good empirical value for the k contrast factor
// * given an input image, the percentile (0-1), the gradient scale and the number of
// * bins in the histogram
// * @param img Input image
// * @param perc Percentile of the image gradient histogram (0-1)
// * @param gscale Scale for computing the image gradient histogram
// * @param nbins Number of histogram bins
// * @param ksize_x Kernel size in X-direction (horizontal) for the Gaussian smoothing kernel
// * @param ksize_y Kernel size in Y-direction (vertical) for the Gaussian smoothing kernel
// * @return k contrast factor
// */
//float Compute_K_Percentile(const cv::Mat &img, float perc, float gscale,
//                           size_t nbins, size_t ksize_x, size_t ksize_y);
///**
// * @brief This function computes Scharr image derivatives
// * @param src Input image
// * @param dst Output image
// * @param xorder Derivative order in X-direction (horizontal)
// * @param yorder Derivative order in Y-direction (vertical)
// * @param scale Scale factor for the derivative size
// */
//void Compute_Scharr_Derivatives(const cv::Mat &src, cv::Mat &dst,
//                                const int xorder, const int yorder,
//                                const int scale);
///**
// * @brief This function performs a scalar non-linear diffusion step
// * @param Ld2 Output image in the evolution
// * @param c Conductivity image
// * @param Lstep Previous image in the evolution
// * @param stepsize The step size in time units
// * @note Forward Euler Scheme 3x3 stencil
// * The function c is a scalar value that depends on the gradient norm
// * dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
// */
//void NLD_Step_Scalar(cv::Mat &Lt, const cv::Mat &c, cv::Mat &Lstep,
//                     float stepsize);
///**
// * @brief This function downsamples the input image with the kernel [1/4,1/2,1/4]
// * @param img Input image to be downsampled
// * @param dst Output image with half of the resolution of the input image
// */
//void Downsample_Image(const cv::Mat &src, cv::Mat &dst);
///**
// * @brief This function downsamples the input image using OpenCV resize
// * @param img Input image to be downsampled
// * @param dst Output image with half of the resolution of the input image
// */
//void Halfsample_Image(const cv::Mat &src, cv::Mat &dst);
///**
// * @brief Compute Scharr derivative kernels for sizes different than 3
// * @param kx_ The derivative kernel in x-direction
// * @param ky_ The derivative kernel in y-direction
// * @param dx The derivative order in x-direction
// * @param dy The derivative order in y-direction
// * @param scale_ The kernel size
// */
//void Compute_Deriv_Kernels(cv::OutputArray &kx_, cv::OutputArray &ky_,
//                           const int dx, const int dy, const int scale_);

} /* namespace AKAZE */
} /* namespace DO */

#endif /* AKAZE_NONLINEAR_DIFFUSION_HPP */