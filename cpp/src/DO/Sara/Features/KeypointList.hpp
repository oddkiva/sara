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

#include <Eigen/StdVector>

#include <DO/Sara/Core/StdVectorHelpers.hpp>
#include <DO/Sara/Core/Numpy.hpp>

#include <DO/Sara/Features/Feature.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup feature_types

    @{

  */

  template <typename T>
  using DescriptorMatrix = Tensor_<T, 2>;

  template <typename F, typename T>
  using KeypointList = std::tuple<std::vector<F>, Tensor_<T, 2>>;


  template <typename F, typename T>
  inline auto features(const KeypointList<F, T>& keys) -> const std::vector<F>&
  {
    return std::get<0>(keys);
  }

  template <typename F, typename T>
  inline auto features(KeypointList<F, T>& keys) -> std::vector<F>&
  {
    return std::get<0>(keys);
  }

  template <typename F, typename T>
  inline auto descriptors(const KeypointList<F, T>& keys) -> const Tensor_<T, 2>&
  {
    return std::get<1>(keys);
  }

  template <typename F, typename T>
  inline auto descriptors(KeypointList<F, T>& keys) -> Tensor_<T, 2>&
  {
    return std::get<1>(keys);
  }

  template <typename F, typename T>
  inline auto size(const KeypointList<F, T>& keys)
  {
    return descriptors(keys).rows();
  }

  template <typename F, typename T>
  inline auto resize(const KeypointList<F, T>& keys, int num, int dim)
  {
    auto& [f, d] = keys;
    f.resize(num);
    d.resize(num, dim);
  }

  template <typename F, typename T>
  inline auto stack(const KeypointList<F, T>& keys1, const KeypointList<F, T>& keys2)
  {
    const auto& [f1, d1] = keys1;
    const auto& [f2, d2] = keys2;

    return std::make_tuple(::append(f1, f2), vstack(d1, d2));
  }



  template <typename F, typename T>
  inline auto size_consistency_predicate(const KeypointList<F, T>& keys)
  {
    return int(features(keys).size()) == descriptors(keys).rows();
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */
