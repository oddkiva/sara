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
#include <iostream>
#include <fstream>
#include <sstream>

namespace DO {

  template bool readKeypoints<float>(
    std::vector<OERegion>& features,
    DescriptorMatrix<float>& descriptors,
    const std::string& name);

  template bool readKeypoints<unsigned char>(
    std::vector<OERegion>& features,
    DescriptorMatrix<unsigned char>& descriptors,
    const std::string& name);

  template
  bool writeKeypoints<float>(
    const std::vector<OERegion>& features,
    const DescriptorMatrix<float>& descriptors,
    const std::string& name);

  template
  bool writeKeypoints<unsigned char>(
    const std::vector<OERegion>& features,
    const DescriptorMatrix<unsigned char>& descriptors,
    const std::string& name);

}