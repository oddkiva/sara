// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#pragma once

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/Graphics.hpp>

namespace DO::Sara {

  //! @brief Mikolajczyk dataset structure.
  class DO_SARA_EXPORT MikolajczykDataset
  {
  public:
    MikolajczykDataset(const std::string& parentFolderPath,
                       const std::string& name)
      : _parent_folder_path(parentFolderPath)
      , _name(name)
    {
      load_images();
      load_ground_truth_homographies();
    }

    bool load_keys(const std::string& feature_type);
    void check() const;

    const std::string& feature_type() const
    {
      return _feat_type;
    }

    const std::string& name() const
    {
      return _name;
    }

    std::string folder_path() const
    {
      return _parent_folder_path + "/" + _name;
    }

    const Image<Rgb8>& image(size_t i) const
    {
      return _image[i];
    }

    const Matrix3f& H(size_t i) const
    {
      return _H[i];
    }

    const KeypointList<OERegion, float>& keys(size_t i) const
    {
      return _keys[i];
    }

  private:
    bool load_images();
    bool load_ground_truth_homographies();

  private:
    std::string _parent_folder_path;
    std::string _name;
    std::string _feat_type;
    std::vector<Image<Rgb8>> _image;
    std::vector<Matrix3f> _H;
    std::vector<KeypointList<OERegion, float>> _keys;
  };

} /* namespace DO::Sara */
