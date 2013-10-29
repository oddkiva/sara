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

#include <DO/Features.hpp>
#include <DO/Graphics.hpp>

//#define DEBUG_REDUNDANCIES

using namespace std;

namespace DO {

  template struct EqualDescriptor<unsigned char>;
  template struct CompareFeatures<unsigned char>;
  template struct CompareFeatures<float>;
  

  void removeRedundancies(vector<OERegion>& features,
                          DescriptorMatrix<float>& descriptors)
  { 
    if (features.size() != descriptors.size())
    {
      cerr << "Fatal: number of features and descriptors are not equal" << endl;
      throw 0;
    }

    vector<int> indices(features.size());
    for (int i = 0; i < indices.size(); ++i)
      indices[i] = i;
    CompareFeatures<float> compareDescriptors(features, descriptors);
    sort(indices.begin(), indices.end(), compareDescriptors);

#ifdef DEBUG_LEXICOGRAPHICAL_ORDER
    for (int i = 0; i < indices.size(); ++i)
    {
      cout << descriptors[indices[i]].transpose() << endl << endl;
      getKey();
    }
#endif

    EqualDescriptor<float> equalDescriptors(descriptors);
    for (int i = 0; i != indices.size(); )
    {
#ifdef DEBUG_REDUNDANCIES
      features[indices[i]].draw(Green8);
      cout << features[indices[i]] << endl;
#endif
      int num;
      for (num = 1;
           num != indices.size()-i && equalDescriptors(indices[i], indices[i+num]);
           ++num)
      {
#ifdef DEBUG_REDUNDANCIES
        features[indices[i+num]].draw(Red8);
        cout << features[indices[i+num]] << endl;
#endif
      }
      i += num;
#ifdef DEBUG_REDUNDANCIES
      if (num > 1)
      {
        cout << "redundant = " << num << endl;
        getKey();
      }
#endif
    }

    indices.resize(
      unique(indices.begin(), indices.end(), equalDescriptors)
      - indices.begin() );
#ifdef DEBUG_REDUNDANCIES
    cout << indices.size() << endl;
#endif

    vector<OERegion> features2(indices.size());
    DescriptorMatrix<float> descriptors2(
      int(indices.size()), descriptors.dimension() );

    for (int i = 0; i < indices.size(); ++i)
    {
      features2[i] = features[indices[i]];
      descriptors2[i] = descriptors[indices[i]];
    }
    features.swap(features2);
    descriptors.swap(descriptors2);

#ifdef DEBUG_REDUNDANCIES
#define CHECK(x) cout << #x << " = " << x << endl
    CHECK(features.size());
    CHECK(descriptors.size());
#endif
  }

  
} /* namespace DO */