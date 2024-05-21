// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/SfM/Helpers/EssentialMatrixEstimation.hpp>

#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/ErrorMeasures.hpp>
#include <DO/Sara/MultiViewGeometry/MinimalSolvers/EssentialMatrixSolvers.hpp>
#include <DO/Sara/RANSAC/RANSAC.hpp>
#include <DO/Sara/SfM/Helpers/FundamentalMatrixEstimation.hpp>


namespace DO::Sara {

  using ESolver = NisterFivePointAlgorithm;

  auto estimate_essential_matrix(const std::vector<Match>& Mij,            //
                                 const KeypointList<OERegion, float>& ki,  //
                                 const KeypointList<OERegion, float>& kj,  //
                                 const Eigen::Matrix3d& Ki_inv,            //
                                 const Eigen::Matrix3d& Kj_inv,            //
                                 int num_samples,                          //
                                 double err_thres)
      -> std::tuple<EssentialMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>
  {
    const auto& fi = features(ki);
    const auto& fj = features(kj);
    const auto ui = extract_centers(fi).cast<double>();
    const auto uj = extract_centers(fj).cast<double>();

    const auto uni = apply_transform(Ki_inv, homogeneous(ui));
    const auto unj = apply_transform(Kj_inv, homogeneous(uj));

    const auto Mij_tensor = to_tensor(Mij);
    const auto Xij = PointCorrespondenceList{Mij_tensor, uni, unj};

    auto inlier_predicate = InlierPredicate<AlgebraicEpipolarDistance>{};
    inlier_predicate.err_threshold = err_thres;

    const auto [E, inliers, sample_best] =
        ransac(Xij, ESolver{}, inlier_predicate, num_samples);

    SARA_CHECK(E);
    SARA_CHECK(inliers.row_vector());
    SARA_CHECK(Mij.size());

    return std::make_tuple(E, inliers, sample_best);
  }

} /* namespace DO::Sara */
