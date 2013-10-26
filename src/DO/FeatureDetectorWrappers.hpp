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

#ifndef DO_FEATUREDETECTORWRAPPERS_HPP
#define DO_FEATUREDETECTORWRAPPERS_HPP

#include <DO/Features.hpp>

// Third-party software wrappers
//
// Affine-covariant features detected by Mikolajczyk's binary.
#include "FeatureDetectorWrappers/HarAffSiftDetector.hpp"
#include "FeatureDetectorWrappers/HesAffSiftDetector.hpp"
#include "FeatureDetectorWrappers/MserSiftDetector.hpp"

#ifdef EXTERNBINDIR
# define EXTERNBINDIR_STRINGIFY(s)  #s
# define EXTERNBINDIR_EVAL(s) EXTERNBINDIR_STRINGIFY(s)
# define externBinPath(s) (EXTERNBINDIR_EVAL(EXTERNBINDIR)"/"s)  
#endif

//! \defgroup FeatureDetectors


#endif /* DO_FEATUREDETECTORWRAPPERS_HPP */