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

#include <DO/Sara/Defines.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/Features/KeypointList.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>



namespace DO { namespace Sara {

  /*!
    @ingroup Features
    @{
  */

  template <typename T>
  bool read_keypoints(std::vector<OERegion>& features,
                      Tensor_<T, 2>& descriptors,
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

    using RowVectorXT = Matrix<T, 1, Dynamic>;
    auto descriptor_i = RowVectorXT{descriptor_dim};

    auto dmat = descriptors.matrix();
    for (int i = 0; i < num_features; ++i)
    {
      file >> features[i];
      file >> descriptor_i;
      dmat.row(i) = descriptor_i;
    }

    file.close();

    return true;
  }

  template <typename T>
  bool write_keypoints(const std::vector<OERegion>& features,
                       const TensorView_<T, 2>& descriptors,
                       const std::string& name)
  {
    using namespace std;
    ofstream file{name.c_str()};
    if (!file.is_open())
    {
      cerr << "Can't open file" << std::endl;
      return false;
    }

    file << features.size() << " " << descriptors.cols() << std::endl;
    for(size_t i = 0; i < features.size(); ++i)
    {
      const OERegion& feat = features[i];

      file << feat.x() << ' ' << feat.y() << ' ';
      file << Map<const RowVector4f>(feat.shape_matrix.data()) << ' ';
      file << feat.orientation << ' ';
      file << int(feat.type) << ' ';

      file << Map<const Matrix<T, 1, Dynamic> >(
        descriptors[static_cast<int>(i)].data(),
        1, descriptors.cols() ) << endl;
    }

    file.close();

    return true;
  }

  inline auto read_keypoints(H5File& h5_file, const std::string& group_name)
      -> KeypointList<OERegion, float>
  {
    auto features = std::vector<OERegion>{};
    auto descriptors = Tensor_<float, 2>{};

    h5_file.read_dataset(group_name + "/" + "features", features);
    h5_file.read_dataset(group_name + "/" + "descriptors", descriptors);

    return {features, descriptors};
  }

  inline auto write_keypoints(H5File& h5_file, const std::string& group_name,
                              const KeypointList<OERegion, float>& keys)
  {
    const auto& [f, v] = keys;
    h5_file.write_dataset(group_name + "/" + "features", tensor_view(f));
    h5_file.write_dataset(group_name + "/" + "descriptors", v);
  }

  //! @}

} /* namespace Sara */
} /* namespace DO */
