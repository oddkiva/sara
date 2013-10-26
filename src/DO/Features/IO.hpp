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

//! @file

#ifndef DO_FEATURES_IO_HPP
#define DO_FEATURES_IO_HPP

namespace DO {

  /*!
    \ingroup Features
    @{
  */

  template <typename T>
  bool readKeypoints(std::vector<OERegion>& features,
                     DescriptorMatrix<T>& descriptors,
                     const std::string& name)
  {
    using namespace std;
    ifstream file(name.c_str());
    if (!file.is_open()) {
      cerr << "Cant open file " << name << endl;    
      return false;
    }
    
    int num_features, descriptor_dimension;
    file >> num_features >> descriptor_dimension;

    cout << "num_features = " << num_features << endl;
    cout << "descriptor_dimension = " << descriptor_dimension << endl;
    
    features.resize(num_features);
    descriptors.resize(num_features, descriptor_dimension);

    double doubleFeatType;
    for (int i = 0; i < num_features; ++i)
    {
      OERegion& feat = features[i];
      file >> feat.coords();
      file >> feat.shapeMat();
      file >> feat.orientation();
      file >> doubleFeatType;
      feat.type() = OERegion::Type(int(doubleFeatType));
      file >> descriptors[i];

      /*cout 
        << feat.coords().transpose() << " " 
        << feat.shapeMat().row(0) << " " << feat.shapeMat().row(1) << " "
        << feat.orientation() << " "
        << int(feat.type()) << endl;
      cout << descriptors[i].transpose() << endl;*/
    }
    file.close();
    return true;
  }

  template <typename T>
  bool writeKeypoints(const std::vector<OERegion>& features,
                      const DescriptorMatrix<T>& descriptors,
                      const std::string& name)
  {
    using namespace std;
    ofstream file(name.c_str());
    if (!file.is_open()) {
      cerr << "Cant open file" << std::endl;    
      return false;
    }

    file << features.size() << " " << descriptors.dimension() << std::endl;
    for(size_t i = 0; i < features.size(); ++i)
    {
      const OERegion& feat = features[i];

      file << feat.x() << ' ' << feat.y() << ' ';
      file << Map<const RowVector4f>(feat.shapeMat().data()) << ' ';
      file << feat.orientation() << ' ';
      file << double(feat.type()) << ' ';
      
      file << Map<const Matrix<T, 1, Dynamic> >(descriptors[static_cast<int>(i)].data(), 1, descriptors.dimension()) << endl;
    }
    file.close();
    return true;
  }

  //! @}

} /* namespace DO */

#endif /* DO_FEATURES_IO_HPP */
