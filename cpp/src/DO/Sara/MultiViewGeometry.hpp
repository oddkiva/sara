// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

#include <DO/Sara/MultiViewGeometry/Datasets/Strecha.hpp>

#include <DO/Sara/MultiViewGeometry/Geometry/EssentialMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/FundamentalMatrix.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Homography.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/Normalizer.hpp>
#include <DO/Sara/MultiViewGeometry/Geometry/TwoViewGeometry.hpp>

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/FivePointAlgoRefImpls.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/EightPointAlgorithm.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/HomographySolver.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/SevenPointAlgorithm.hpp>

#include <DO/Sara/MultiViewGeometry/HDF5.hpp>

//! @defgroup MultiViewGeometry Multi-View Geometry
