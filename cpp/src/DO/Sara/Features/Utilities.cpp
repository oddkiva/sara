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

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <DO/Sara/Features.hpp>

#include <DO/Sara/Graphics.hpp>


using namespace std;


namespace DO { namespace Sara {


  void remove_redundant_features(vector<OERegion>& features,
                                 Tensor_<float, 2>& descriptors)
  {
    if (features.size() != descriptors.size())
      throw std::runtime_error{
        "Fatal: the number of features and descriptors are not equal"
      };

    // Equality lambda functor.
    auto compare_equal = [&](size_t i1, size_t i2) {
      return (descriptors.matrix().row(i1) - descriptors.matrix().row(i2))
                 .squaredNorm() < 1e-3;
    };

    auto dmat = descriptors.matrix();

    // Lexicographical comparison lambda functor.
    auto compare_less = [&](size_t i1, size_t i2) {
      if (Sara::lexicographical_compare(dmat.row(i1), dmat.row(i2)))
        return true;
      if (compare_equal(i1, i2) &&
          features[i1].extremum_value > features[i2].extremum_value)
        return true;
      return false;
    };

    // Remove redundant features.
    //
    // Sort.
    auto indices = vector<size_t>(features.size());
    for (size_t i = 0; i < indices.size(); ++i)
      indices[i] = i;
    sort(indices.begin(), indices.end(), compare_less);

    // Remove duplicates.
    auto it = unique(indices.begin(), indices.end(), compare_equal);
    indices.resize(it - indices.begin());

    auto unique_features = vector<OERegion>(indices.size());
    auto unique_descriptors =
        Tensor_<float, 2>{{indices.size(), descriptors.size(1)}};

    auto unique_dmat = unique_descriptors.matrix();

    for (size_t i = 0; i < indices.size(); ++i)
    {
      unique_features[i] = features[indices[i]];
      unique_dmat.row(i) = dmat.row(indices[i]);
    }

    // Swap data.
    features.swap(unique_features);
    descriptors.swap(unique_descriptors);
  }


} /* namespace Sara */
} /* namespace DO */
