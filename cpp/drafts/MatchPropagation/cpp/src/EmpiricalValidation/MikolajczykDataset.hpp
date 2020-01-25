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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Features.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>

namespace DO::Sara {

  class MikolajczykDataset
  {
  public:
    MikolajczykDataset(const std::string& parentFolderPath,
                       const std::string& name)
      : parent_folder_path_(parentFolderPath)
      , name_(name)
    {
      loadImages();
      loadGroundTruthHs();
    }

    bool loadKeys(const std::string& featType);
    void check() const;

    const std::string& featType() const { return feat_type_; }
    const std::string& name() const { return name_; }
    std::string folderPath() const { return parent_folder_path_+"/"+name_; }
    const Image<Rgb8>& image(size_t i) const { return image_[i]; }
    const Matrix3f& H(size_t i) const { return H_[i]; }
    const Set<OERegion, RealDescriptor>& keys(size_t i) const { return keys_[i]; }

  private:
    bool loadImages();
    bool loadGroundTruthHs();

  private:
    std::string parent_folder_path_;
    std::string name_;
    std::string feat_type_;
    std::vector<Image<Rgb8> > image_;
    std::vector<Matrix3f> H_;
    std::vector<Set<OERegion, RealDescriptor> > keys_;
  };

} /* namespace DO::Sara */
