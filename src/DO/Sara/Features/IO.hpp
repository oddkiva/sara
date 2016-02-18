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

#ifndef DO_SARA_FEATURES_IO_HPP
#define DO_SARA_FEATURES_IO_HPP


#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/Features/DescriptorMatrix.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Features
    @{
  */

  template <typename T>
  bool read_keypoints(std::vector<OERegion>& features,
                      DescriptorMatrix<T>& descriptors,
                      const std::string& name)
  {
    using namespace std;
    ifstream file{ name.c_str() };
    if (!file.is_open())
    {
      cerr << "Can't open file " << name << endl;
      return false;
    }

    int num_features, descriptor_dim;
    file >> num_features >> descriptor_dim;

    features.resize(num_features);
    descriptors.resize(num_features, descriptor_dim);
    Matrix<T, Dynamic, 1> descriptor_i(descriptor_dim, 1);

    for (int i = 0; i < num_features; ++i)
    {
      file >> features[i];
      file >> descriptor_i;
      descriptors[i] = descriptor_i;
    }

    file.close();

    return true;
  }

  template <typename T>
  bool write_keypoints(const std::vector<OERegion>& features,
                       const DescriptorMatrix<T>& descriptors,
                       const std::string& name)
  {
    using namespace std;
    ofstream file{ name.c_str() };
    if (!file.is_open())
    {
      cerr << "Can't open file" << std::endl;
      return false;
    }

    file << features.size() << " " << descriptors.dimension() << std::endl;
    for(size_t i = 0; i < features.size(); ++i)
    {
      const OERegion& feat = features[i];

      file << feat.x() << ' ' << feat.y() << ' ';
      file << Map<const RowVector4f>(feat.shape_matrix().data()) << ' ';
      file << feat.orientation() << ' ';
      file << int(feat.type()) << ' ';

      file << Map<const Matrix<T, 1, Dynamic> >(
        descriptors[static_cast<int>(i)].data(),
        1, descriptors.dimension() ) << endl;
    }

    file.close();

    return true;
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_FEATURES_IO_HPP */
