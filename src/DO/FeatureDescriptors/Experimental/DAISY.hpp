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

#ifndef DO_FEATUREDESCRIPTORS_DAISY_HPP
#define DO_FEATUREDESCRIPTORS_DAISY_HPP

class daisy;

namespace DO {

  class DAISY {
  public:
    DAISY();
    ~DAISY();
    void compute(VectorXf& descriptor, float x, float y, float o) const;
    void compute(DescriptorMatrix<float>& daisies,
                 const std::vector<OERegion>& features,
                 const std::vector<Vector2i>& scaleOctPairs,
                 const ImagePyramid<float>& pyramid) const;
    void initialize(const Image<float>& image) const;
    void reset() const;
    int dimension() const;

  private:
    daisy *daisy_computer_;
  };

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDESCRIPTORS_DAISY_HPP */
