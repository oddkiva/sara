// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/FileSystem.hpp>
#include <iostream>
#include <boost/program_options.hpp>

using namespace DO;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
  string in;

  // Parser
  po::command_line_parser(argc, argv);
  // Description
  po::options_description desc("DOFileSystemTest");
  // Command options
  desc.add_options()
    ("help,h", "Produce help message")
    ("in,i", po::value<string>(&in), "Input directory")
    ;
  // Required options.
  po::positional_options_description p;
  p.add("in", 1);

  // Parse options.
  try
  {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
      options(desc).
      positional(p).
      run(),
      vm);
    notify(vm);
    if(vm.count("help") || !vm.count("in"))
      cout << desc << endl; 
  }
  catch(const std::exception& e)
  {
    cout << e.what() << endl;
  }


  // Get image filenames.
  vector<string> filenames;
  getImageFilePaths(filenames, in);

  
	return 0;
}