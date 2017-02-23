// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "MikolajczykDataset.hpp"

using namespace std;

namespace DO {

  bool MikolajczykDataset::loadKeys(const string& featType)
  {
    feat_type_ = featType;
    printStage("Loading Keypoints: "+ featType + " for Dataset: " + name_);
    keys_.resize(image_.size());
    for (size_t i = 0; i != keys_.size(); ++i)
    {
      if (!readKeypoints(keys_[i].features, keys_[i].descriptors,
                         folderPath()+"/img"+toString(i+1)+featType))
        return false;
    }
    return true;
  }

  void MikolajczykDataset::check() const
  {
    printStage("Checking images");
    openWindow(image_.front().width(), image_.front().height());
    for (int i = 0; i < image_.size(); ++i)
    {
      display(image_[i]);
      getKey();
    }
    closeWindow();

    printStage("Checking ground truth homographies");
    for (int i = 0; i < 6; ++i)
      cout << "H[" << i << "]=\n" << H_[i] << endl;
  }

  bool MikolajczykDataset::loadImages()
  {
    printStage("Loading images of Dataset: "+name_);
    image_.resize(6);
    for (int i = 0; i < 6; ++i)
    {
      string path = folderPath() + "/img" + toString(i+1) + ".ppm";
      string path2 = folderPath() + "/img" + toString(i+1) + ".pgm";
      bool readOk = load(image_[i], path) || load(image_[i], path2);
      if ( !readOk )
      {
        cerr << "Error: could not load image from path:\n" << path << endl;
        return false;
      }
    }
    return true;
  }

  bool MikolajczykDataset::loadGroundTruthHs()
  {
    printStage("Loading ground truth homographies of Dataset: "+name_);
    H_.resize(6);
    for (int i = 0; i < 6; ++i)
    {
      if (i == 0)
      {
        H_[i].setZero();
        continue;
      }

      string path = folderPath() + "/H1to" + toString(i+1) + "p";
      ifstream f(path.c_str());
      if (!f.is_open())
      {
        cerr << "Error: could not load ground truth homography from path:\n"
             << path << endl;
        return false;
      }

      f >> H_[i];
    }
    return true;
  }

} /* namespace DO */