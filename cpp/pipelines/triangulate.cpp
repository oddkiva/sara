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

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/Datasets/Strecha.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;


void triangulate(const std::string& dirpath, const std::string& h5_filepath)
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    sara::cp(h5_filepath, h5_filepath + ".bak");

  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDWR};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, sara::ls(dirpath, ".png"));
  append(image_paths, sara::ls(dirpath, ".jpg"));
  std::sort(image_paths.begin(), image_paths.end());

  auto group_names = std::vector<std::string>{};
  group_names.reserve(image_paths.size());
  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::back_inserter(group_names),
                 [&](const std::string& image_path) {
                   return sara::basename(image_path);
                 });
}


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Estimate essential matrices"};
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
    triangulate(dirpath, h5_filepath);

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
