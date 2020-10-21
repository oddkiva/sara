// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
//! @brief Master header file of the ImageProcessing module.

#pragma once

#ifdef _OPENMP
# include <omp.h>
#endif

#include <vector>
#include <exception>

#include <DO/Sara/Core.hpp>

// Basic image processing functions.
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/ImageProcessing/Deriche.hpp>

// GEMM-based convolution.
#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

// Gradient, Laplacian, Hessian matrix, norm, orientation, second moment matrix.
#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/Determinant.hpp>
#include <DO/Sara/ImageProcessing/Norm.hpp>
#include <DO/Sara/ImageProcessing/Orientation.hpp>
#include <DO/Sara/ImageProcessing/SecondMomentMatrix.hpp>

// Interpolation (bilinear, trilinear).
#include <DO/Sara/ImageProcessing/Interpolation.hpp>

// Flip, reduce, enlarge, downscale, upscale.
#include <DO/Sara/ImageProcessing/Flip.hpp>
#include <DO/Sara/ImageProcessing/Resize.hpp>
#include <DO/Sara/ImageProcessing/Warp.hpp>

// Data structures and functions for feature detections.
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>
#include <DO/Sara/ImageProcessing/Extrema.hpp>
#include <DO/Sara/ImageProcessing/GaussianPyramid.hpp>

// Color perturbations.
#include <DO/Sara/ImageProcessing/ColorFancyPCA.hpp>
#include <DO/Sara/ImageProcessing/ColorJitter.hpp>
#include <DO/Sara/ImageProcessing/ColorStatistics.hpp>

// Data augmentation.
#include <DO/Sara/ImageProcessing/DataAugmentation.hpp>

// Edge detection.
#include <DO/Sara/ImageProcessing/EdgeDetection.hpp>

// Watershed.
#include <DO/Sara/ImageProcessing/Watershed.hpp>

/*!
  @defgroup ImageProcessing Image Processing
  @brief The Image Processing module is a header-only library.
  It covers basic image processing features such as:
  - Linear filtering (both separable and non-separable)
  - Gaussian blurring
  - Deriche IIR filters
  - Image enlarging/reducing functions
  - Interpolation
  - Differential calculus (gradient, Hessian matrix, divergence, Laplacian)
    based on central finite differentiation.
  - Second-moment matrix
  - Image Pyramid data structure
  - Difference of Gaussians computation
  - Local extremum localization, including localization in scale-space.
 */
