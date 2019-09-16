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

#include <DO/Sara/Features.hpp>

#include <iostream>
#include <fstream>
#include <sstream>


namespace DO { namespace Sara {

  template bool read_keypoints<float>(std::vector<OERegion>& features,
                                      Tensor_<float, 2>& descriptors,
                                      const std::string& name);

  template bool
  read_keypoints<std::uint8_t>(std::vector<OERegion>& features,
                               Tensor_<std::uint8_t, 2>& descriptors,
                               const std::string& name);

  template bool write_keypoints<float>(const std::vector<OERegion>& features,
                                       const TensorView_<float, 2>& descriptors,
                                       const std::string& name);

  template bool
  write_keypoints<std::uint8_t>(const std::vector<OERegion>& features,
                                const TensorView_<std::uint8_t, 2>& descriptors,
                                const std::string& name);

} /* namespace Sara */
} /* namespace DO */
