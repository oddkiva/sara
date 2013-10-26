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

#ifndef DO_FEATUREDETECTORS_HPP
#define DO_FEATUREDETECTORS_HPP

#include <DO/Defines.hpp>
#include <DO/Graphics.hpp>
#include <DO/Features.hpp>
#include <DO/ImageProcessing.hpp>

// Utilities and debug.
#include "FeatureDetectors/Debug.hpp"

// Extremum filtering and refining.
#include "FeatureDetectors/RefineExtremum.hpp"

// Feature detection.
#include "FeatureDetectors/DoG.hpp"
#include "FeatureDetectors/Harris.hpp"
#include "FeatureDetectors/Hessian.hpp"
//#include "FeatureDetectors/MSER.hpp"

// Affine shape adaptation (cf. [Mikolajczyk & Schmid, ECCV 2002]).
#include "FeatureDetectors/AffineShapeAdaptation.hpp"

//! \defgroup FeatureDetectors


#endif /* DO_FEATUREDETECTORS_HPP */