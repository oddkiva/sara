// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2018-2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/Numpy.hpp>
#include <DO/Sara/MultiViewGeometry/Estimators/RelativePoseEstimator.hpp>


namespace DO::Sara {

template <typename Method>
auto RelativePoseEstimator<Method>::operator()(
    const Matrix<double, 3, RelativePoseEstimator<Method>::N>& left,
    const Matrix<double, 3, RelativePoseEstimator<Method>::N>& right) const
    -> std::vector<TwoViewGeometry>
{
  const auto Es = this->Method::find_essential_matrices(left, right);

  auto indices = range(Es.size());

  auto motions = Tensor_<Motion, 2>{{int(Es.size()), 4}};
  std::for_each(std::begin(indices), std::end(indices), [&](int i) {
    motions[i] = tensor_view(extract_relative_motion_horn(Es[i]));
  });

  auto geometries = std::vector<TwoViewGeometry>(motions.size());
  std::transform(
      std::begin(motions), std::end(motions), std::begin(geometries),
      [&](const auto& m) { return two_view_geometry(m, left, right); });

  if (cheiral_criterion == CheiralityCriterion::NONE)
    return geometries;

  const auto best_geom =
      std::max_element(std::begin(geometries), std::end(geometries),
                       [](const auto& g1, const auto& g2) {
                         return g1.cheirality.count() < g2.cheirality.count();
                       });

  const auto cheiral_degree = best_geom->cheirality.count();
  // Remove geometries where all the points violates the cheirality.
  geometries.erase(std::remove_if(std::begin(geometries), std::end(geometries),
                                  [&](const auto& g) {
                                    return g.cheirality.count() !=
                                           cheiral_degree;
                                  }),
                   geometries.end());

  // Filter the estimated camera poses.
  if (cheiral_criterion == CheiralityCriterion::CHEIRAL_MOST)
    return geometries;

  if (cheiral_criterion == CheiralityCriterion::CHEIRAL_COMPLETE)
    return cheiral_degree < N ? std::vector<TwoViewGeometry>{} : geometries;

  throw std::runtime_error{"Not implemented for this cheirality criterion"};
}


template struct DO_SARA_EXPORT RelativePoseEstimator<NisterFivePointAlgorithm>;
template struct DO_SARA_EXPORT
    RelativePoseEstimator<SteweniusFivePointAlgorithm>;

} /* namespace DO::Sara */
