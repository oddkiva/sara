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

#include <DO/Sara/Core/StdVectorHelpers.hpp>
#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <DO/Sara/Features/Draw.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sara = DO::Sara;


void detect_keypoints(const std::string& dirpath,
                      const std::string& h5_filepath)
{
  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_TRUNC};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, sara::ls(dirpath, ".png"));
  append(image_paths, sara::ls(dirpath, ".jpg"));

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = sara::imread<float>(path);

        SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
        const auto keys = sara::compute_sift_keypoints(image);

        const auto group_name = sara::basename(path);
        h5_file.group(group_name);

        SARA_DEBUG << "Saving SIFT keypoints of " << path << "..." << std::endl;
        write_keypoints(h5_file, group_name, keys);
      });
}


void read_keypoints(const std::string& dirpath, const std::string& h5_filepath)
{
  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDONLY};
  auto image_paths = std::vector<std::string>{};
  append(image_paths, sara::ls(dirpath, ".png"));
  append(image_paths, sara::ls(dirpath, ".jpg"));

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = sara::imread<float>(path);

        const auto group_name = sara::basename(path);

        SARA_DEBUG << "Read keypoints for " << group_name << "..." << std::endl;
        const auto keys =
            read_keypoints(h5_file, group_name + "/" + "descriptors");

        const auto& features = std::get<0>(keys);

        // Visual inspection.
        if (!sara::active_window())
        {
          sara::create_window(image.sizes() / 2, group_name);
          sara::set_antialiasing();
        }

        if (sara::get_sizes(sara::active_window()) != image.sizes() / 2)
          sara::resize_window(image.sizes() / 2);

        sara::display(image, 0, 0, 0.5);
        sara::draw_oe_regions(features, sara::Red8, 0.5f);
        sara::get_key();
        sara::close_window();
      });
}


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Detect SIFT keypoints"};
    desc.add_options()                                                 //
        ("help, h", "Help screen")                                     //
        ("dirpath", po::value<std::string>(), "Image directory path")  //
        ("out_h5_file", po::value<std::string>(), "Output HDF5 file")  //
        ("read", "Visualize detected keypoints")  //
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      return 0;
    }

    if (!vm.count("dirpath"))
    {
      std::cout << "Missing image directory path" << std::endl;
      return 0;
    }
    if (!vm.count("out_h5_file"))
    {
      std::cout << desc << std::endl;
      std::cout << "Missing output H5 file path" << std::endl;
      return 0;
    }

    const auto dirpath = vm["dirpath"].as<std::string>();
    const auto h5_filepath = vm["out_h5_file"].as<std::string>();
    if (vm.count("read"))
      read_keypoints(dirpath, h5_filepath);
    else
      detect_keypoints(dirpath, h5_filepath);

    return 0;
  }
  catch (const po::error& e)
  {
    std::cerr << e.what() << "\n";
    return 1;
  }
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
