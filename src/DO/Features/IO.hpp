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

  std::ostream& operator<<(std::ostream& os, const Keypoint& k);

  bool readKeypoints(std::vector<Keypoint>& keys, const std::string & name,
                     bool bundlerFormat = false);  

  bool writeKeypoints(const std::vector<Keypoint>& keys, const std::string & name, 
                      bool writeForBundler = false);

  //! @}

} /* namespace DO */

#endif /* DO_FEATURES_IO_HPP */
