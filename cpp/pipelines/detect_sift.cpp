// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


namespace sara = DO::Sara;


void detect_keypoints()
{
  const auto dirpath = fs::path{"/home/david/Desktop/Datasets/sfm/castle_int"};
  const auto image_paths = sara::ls(dirpath.string(), ".png");

  auto h5_file = sara::H5File{
      "/home/david/Desktop/Datasets/sfm/castle_int.h5", H5F_ACC_TRUNC};

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = sara::imread<float>(path);

        SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
        const auto keys = sara::compute_sift_keypoints(image);

        const auto group_name = sara::basename(path);
        h5_file.group(group_name);

        const auto& [f, v] = keys;

        SARA_DEBUG << "Saving SIFT keypoints of " << path << "..." << std::endl;
        h5_file.write_dataset(group_name + "/" + "features", tensor_view(f));
        h5_file.write_dataset(group_name + "/" + "descriptors", v);
      });

}

void read_keypoints()
{
  const auto dirpath = fs::path{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto image_paths = sara::ls(dirpath.string(), ".png");

  auto h5_file = sara::H5File{"/home/david/Desktop/Datasets/sfm/castle_int.h5",
                              H5F_ACC_RDONLY};

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = sara::imread<float>(path);

        const auto group_name = sara::basename(path);

        auto features = sara::Tensor_<sara::OERegion, 1>{};
        auto descriptors = sara::Tensor_<float, 2>{};

        SARA_DEBUG << "Read DoG features for " << group_name << "..." << std::endl;
        h5_file.read_dataset(group_name + "/" + "features", features);

        SARA_DEBUG << "Read SIFT descriptors for " << group_name << "..." << std::endl;
        h5_file.read_dataset(group_name + "/" + "descriptors", descriptors);


        // Visual inspection.
        if (!sara::active_window())
        {
          sara::create_window(image.sizes() / 2, group_name);
          sara::set_antialiasing();
        }

        if (sara::get_sizes(sara::active_window()) != image.sizes() / 2)
          sara::resize_window(image.sizes() / 2);

        sara::display(image, 0, 0, 0.5);
        sara::draw_oe_regions(features.begin(), features.end(), sara::Red8, 0.5f);
        sara::get_key();
        sara::close_window();
      });
}

GRAPHICS_MAIN()
{
  //detect_keypoints();
  read_keypoints();

  return 0;
}
