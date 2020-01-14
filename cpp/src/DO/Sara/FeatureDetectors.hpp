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

#pragma once

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/ImageProcessing.hpp>

// Utilities and debug.
#include <DO/Sara/FeatureDetectors/Debug.hpp>

// Extremum filtering and refining.
#include <DO/Sara/FeatureDetectors/RefineExtremum.hpp>

// Feature detection.
#include <DO/Sara/FeatureDetectors/LoG.hpp>
#include <DO/Sara/FeatureDetectors/DoG.hpp>
#include <DO/Sara/FeatureDetectors/Harris.hpp>
#include <DO/Sara/FeatureDetectors/Hessian.hpp>
//#include "FeatureDetectors/MSER.hpp"

// Affine shape adaptation (cf. [Mikolajczyk & Schmid, ECCV 2002]).
#include <DO/Sara/FeatureDetectors/AffineShapeAdaptation.hpp>

//! @defgroup FeatureDetectors Feature Detectors
