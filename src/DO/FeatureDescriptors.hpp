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

#ifndef DO_FEATUREDESCRIPTORS_HPP
#define DO_FEATUREDESCRIPTORS_HPP

#include <DO/Defines.hpp>
#include <DO/Graphics.hpp>
#include <DO/Features.hpp>
#include <DO/ImageProcessing.hpp>

// Utilities and debug.
#include "FeatureDetectors/Debug.hpp"

// Feature description.
//
// Assign dominant orientations $\theta$ to image patch $(x,y,\sigma)$ using 
// Lowe's method (cf. [Lowe, IJCV 2004]).
#include "FeatureDescriptors/Orientation.hpp"
// Describe keypoint $(x,y,\sigma,\theta)$ with the SIFT descriptor (cf. 
// [Lowe, IJCV 2004]).
#include "FeatureDescriptors/SIFT.hpp"



//! \defgroup FeatureDescriptors


#endif /* DO_FEATUREDETECTORS_FEATUREDETECTORS_HPP */