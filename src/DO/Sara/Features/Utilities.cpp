// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Features.hpp>
#include <DO/Sara/Graphics.hpp>

//#define DEBUG_REDUNDANCIES


using namespace std;


namespace DO { namespace Sara {

  template struct EqualDescriptor<unsigned char>;
  template struct CompareFeatures<unsigned char>;
  template struct CompareFeatures<float>;


  void remove_redundancies(vector<OERegion>& features,
                           DescriptorMatrix<float>& descriptors)
  {
    if (features.size() != descriptors.size())
      throw std::runtime_error{
        "Fatal: number of features and descriptors are not equal"
      };

    auto indices = vector<size_t>(features.size());
    for (size_t i = 0; i < indices.size(); ++i)
      indices[i] = i;

    CompareFeatures<float> compare_descriptors{ features, descriptors };
    sort(indices.begin(), indices.end(), compare_descriptors);

    EqualDescriptor<float> equal_descriptors{ descriptors };
    auto it = unique(indices.begin(), indices.end(), equal_descriptors);

    indices.resize(it - indices.begin());

    vector<OERegion> unique_features{ indices.size() };
    DescriptorMatrix<float> unique_descriptors{
      indices.size(), descriptors.dimension()
    };

    for (size_t i = 0; i < indices.size(); ++i)
    {
      unique_features[i] = features[indices[i]];
      unique_descriptors[i] = descriptors[indices[i]];
    }

    features.swap(unique_features);
    descriptors.swap(unique_descriptors);
  }


} /* namespace Sara */
} /* namespace DO */
