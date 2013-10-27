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

#ifndef DO_FEATUREDETECTORS_MSERSIFTDETECTOR_HPP
#define DO_FEATUREDETECTORS_MSERSIFTDETECTOR_HPP

namespace DO
{

  /*!
    \ingroup FeatureDetectors
    \defgroup DetectorWrappers
    @{
   */

  //! Mikolajczyk's binaries wrapped for our need.
  class MserSiftDetector
  {
  public:
    MserSiftDetector() {}

    void run(std::vector<OERegion>& features,
             DescriptorMatrix<float>& descriptors,
             const Image<uchar>& I,
             bool specifyThres = false,
             double param = 0) const;

    Set<OERegion, RealDescriptor> run(const Image<uchar>& I, 
                                      bool specifyThres = false,
                                      double HarrisT = 200) const
    {
      Set<OERegion, RealDescriptor> keys;
      run(keys.features, keys.descriptors, I, specifyThres, HarrisT);
      return keys;
    }
  };

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDETECTORS_MSERSIFTDETECTOR_HPP */