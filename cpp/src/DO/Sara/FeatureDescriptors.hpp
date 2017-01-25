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

#ifndef DO_SARA_FEATUREDESCRIPTORS_HPP
#define DO_SARA_FEATUREDESCRIPTORS_HPP

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/ImageProcessing.hpp>

// Utilities and debug.
#include <DO/Sara/FeatureDetectors/Debug.hpp>

// Feature description.
//
// Assign dominant orientations $\theta$ to image patch $(x,y,\sigma)$ using
// Lowe's method (cf. [Lowe, IJCV 2004]).
#include <DO/Sara/FeatureDescriptors/Orientation.hpp>
// Describe keypoint $(x,y,\sigma,\theta)$ with the SIFT descriptor (cf.
// [Lowe, IJCV 2004]).
#include <DO/Sara/FeatureDescriptors/SIFT.hpp>
// Dense feature computation API.
#include <DO/Sara/FeatureDescriptors/DenseFeature.hpp>


//! @defgroup FeatureDescriptors


#endif /* DO_SARA_FEATUREDESCRIPTORS_HPP */