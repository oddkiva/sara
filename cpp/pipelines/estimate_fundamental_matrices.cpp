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

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <iostream>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Estimate fundamental matrices"};
    desc.add_options()                                                 //
        ("help, h", "Help screen")                                     //
        ("dirpath", po::value<std::string>(), "Image directory path")  //
        ("out_h5_file", po::value<std::string>(), "Output HDF5 file")  //
        ("overwrite", "Overwrite fundamental matrices and meta data")  //
        ("debug", "Inspect the fundamental matrix estimation")         //
        ("read", "Read the fundamental matrix estimation")             //
        ("display_step", po::value<int>()->default_value(20),          //
         "Draw correspondences every multiple of the display step")    //
        ("wait_key",                                                   //
         "Wait for key press upon fundamental matrix inspection")      //
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
    const auto overwrite = vm.count("overwrite");
    const auto debug = vm.count("debug");
    const auto read = vm.count("read");
    const auto display_step = vm["display_step"].as<int>();
    const auto wait_key = vm.count("wait_key");

    if (read)
      sara::inspect_fundamental_matrices(dirpath, h5_filepath, display_step,
                                         wait_key);
    else
      sara::estimate_fundamental_matrices(dirpath, h5_filepath, overwrite,
                                          debug);

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
  sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
