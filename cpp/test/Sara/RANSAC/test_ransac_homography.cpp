// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "RANSAC/Homography estimation"

#include <DO/Sara/MultiViewGeometry/MinimalSolvers/HomographySolver.hpp>
#include <DO/Sara/MultiViewGeometry/PointCorrespondenceList.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>

#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_SUITE(TestUtility)

BOOST_AUTO_TEST_CASE(test_concept_requirements)
{
  static_assert(
      sara::DataPointListConcept<sara::PointCorrespondenceList<double>>);
  static_assert(sara::MinimalSolverConcept<sara::FourPointAlgorithm>);
}

BOOST_AUTO_TEST_SUITE_END()
