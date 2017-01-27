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

#include <DO/Sara/Features/Key.hpp>


namespace DO { namespace Sara {

  /*!
    @ingroup Features
    @{
  */

  //! @{
  //! @brief remove redundant features.
  DO_SARA_EXPORT
  void remove_redundant_features(std::vector<OERegion>& features,
                                 DescriptorMatrix<float>& descriptors);

  inline void remove_redundant_features(Set<OERegion, RealDescriptor>& keys)
  {
    remove_redundant_features(keys.features, keys.descriptors);
  }
  //! @}

  //! @}

} /* namespace Sara */
} /* namespace DO */
