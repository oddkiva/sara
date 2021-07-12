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

#include <DO/Sara.hpp>
#include <DO/Sara/SfM/BuildingBlocks/KeypointDetection.hpp>
#include <DO/Sara/Visualization.hpp>


namespace DO::Sara {

auto detect_keypoints(const std::string& dirpath,
                      const std::string& h5_filepath,  //
                      bool overwrite) -> void
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_TRUNC};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = imread<float>(path);

        SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
        const auto keys = compute_sift_keypoints(image);

        const auto group_name = basename(path);
        h5_file.get_group(group_name);

        SARA_DEBUG << "Saving SIFT keypoints of " << path << "..." << std::endl;
        write_keypoints(h5_file, group_name, keys, overwrite);
      });
}


auto read_keypoints(const std::string& dirpath, const std::string& h5_filepath)
    -> void
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};
  auto image_paths = std::vector<std::string>{};
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = imread<float>(path);

        const auto group_name = basename(path);

        SARA_DEBUG << "Read keypoints for " << group_name << "..." << std::endl;
        const auto keys = read_keypoints(h5_file, group_name);

        const auto& features = std::get<0>(keys);

        // Visual inspection.
        if (!active_window())
        {
          create_window(image.sizes() / 2, group_name);
          set_antialiasing();
        }

        if (get_sizes(active_window()) != image.sizes() / 2)
          resize_window(image.sizes() / 2);

        display(image, 0, 0, 0.5);
        draw_oe_regions(features, Red8, 0.5f);
        get_key();
      });

  if (active_window())
    close_window();
}

} /* namespace DO::Sara */
