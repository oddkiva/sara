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
//! @brief Master header file of the ImageProcessing module.

#ifndef DO_SARA_IMAGEPROCESSING_HPP
#define DO_SARA_IMAGEPROCESSING_HPP

#ifdef _OPENMP
# include <omp.h>
#endif
#include <DO/Sara/Core.hpp>

#include <vector>
#include <exception>

// Basic image processing functions.
#include <DO/Sara/ImageProcessing/LinearFiltering.hpp>
#include <DO/Sara/ImageProcessing/Deriche.hpp>
// Gradient, Laplacian, Hessian matrix, norm, orientation, second moment matrix.
#include <DO/Sara/ImageProcessing/Differential.hpp>
#include <DO/Sara/ImageProcessing/Determinant.hpp>
#include <DO/Sara/ImageProcessing/Norm.hpp>
#include <DO/Sara/ImageProcessing/Orientation.hpp>
#include <DO/Sara/ImageProcessing/SecondMomentMatrix.hpp>
// Interpolation (bilinear, trilinear)
#include <DO/Sara/ImageProcessing/Interpolation.hpp>
// Reduce, enlarge, downscale, upscale,
#include <DO/Sara/ImageProcessing/Scaling.hpp>
#include <DO/Sara/ImageProcessing/Warp.hpp>
// Data structures and functions for feature detections.
#include <DO/Sara/ImageProcessing/ImagePyramid.hpp>
#include <DO/Sara/ImageProcessing/Extrema.hpp>
#include <DO/Sara/ImageProcessing/GaussianPyramid.hpp>

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


#endif /* DO_SARA_IMAGEPROCESSING_HPP */