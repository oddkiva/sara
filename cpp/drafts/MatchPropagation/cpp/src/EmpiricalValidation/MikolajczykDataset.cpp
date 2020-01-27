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

#include "MikolajczykDataset.hpp"

using namespace std;

namespace DO::Sara {

  bool MikolajczykDataset::load_keys(const string& feature_type)
  {
    _feat_type = feature_type;
    print_stage("Loading keypoints: " + feature_type + " for Dataset: " + _name);
    _keys.resize(_image.size());
    for (size_t i = 0; i != _keys.size(); ++i)
    {
      if (!read_keypoints(features(_keys[i]), descriptors(_keys[i]),
                          folder_path() + "/img" + to_string(i + 1) + feature_type))
        return false;
    }
    return true;
  }

  void MikolajczykDataset::check() const
  {
    print_stage("Checking images");
    create_window(_image.front().width(), _image.front().height());
    for (int i = 0; i < _image.size(); ++i)
    {
      display(_image[i]);
      get_key();
    }
    close_window();

    print_stage("Checking ground truth homographies");
    for (int i = 0; i < 6; ++i)
      cout << "H[" << i << "]=\n" << _H[i] << endl;
  }

  bool MikolajczykDataset::load_images()
  {
    print_stage("Loading images of Dataset: " + _name);
    _image.resize(6);
    for (int i = 0; i < 6; ++i)
    {
      const auto path = folder_path() + "/img" + to_string(i + 1) + ".ppm";
      const auto path2 = folder_path() + "/img" + to_string(i + 1) + ".pgm";
      const auto read = load(_image[i], path) || load(_image[i], path2);
      if (!read)
      {
        cerr << "Error: could not load image from path:\n" << path << endl;
        return false;
      }
    }
    return true;
  }

  bool MikolajczykDataset::load_ground_truth_homographies()
  {
    print_stage("Loading ground truth homographies of Dataset: " + _name);
    _H.resize(6);
    for (int i = 0; i < 6; ++i)
    {
      if (i == 0)
      {
        _H[i].setZero();
        continue;
      }

      string path = folder_path() + "/H1to" + to_string(i + 1) + "p";
      ifstream f(path.c_str());
      if (!f.is_open())
      {
        cerr << "Error: could not load ground truth homography from path:\n"
             << path << endl;
        return false;
      }

      f >> _H[i];
    }
    return true;
  }

}  // namespace DO::Sara
