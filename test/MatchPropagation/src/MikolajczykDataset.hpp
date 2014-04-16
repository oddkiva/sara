// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#ifndef DO_MIKOLAJCZYK_DATASET_HPP
#define DO_MIKOLAJCZYK_DATASET_HPP

#include <DO/Graphics.hpp>
#include <DO/Features.hpp>
#include <DO/FeatureDetectors.hpp>
#include <DO/FeatureMatching.hpp>

namespace DO {

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

} /* namespace DO */

#endif /* DO_MIKOLAJCZYK_DATASET_HPP */