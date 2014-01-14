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

#include <DO/Core.hpp>
#include <DO/FileSystem.hpp>
#include <iostream>
#include <cmdLine/cmdLine.h>

using namespace DO;
using namespace std;

void print_usage(char **argv)
{
  cout << "Usage: " << argv[0] << " -i input_folder" << endl;
}

int main(int argc, char **argv)
{
  string in;

//  // Create command line options.
//  CmdLine cmdLine;
//  cmdLine.add(make_switch('h', "help"));
//  // Verbose option for debug.
//  cmdLine.add(make_option('i', in, "input folder"));
//  // Image file name.
//  
//  // Try to process
//  try
//  {
//    if (argc == 1)
//      throw std::string("Invalid command line parameter.");
//    
//    cmdLine.process(argc, argv);
//    
//    if (!cmdLine.used('i'))
//      throw std::string("Invalid command line parameter.");
//    
//    if (cmdLine.used('h'))
//      print_usage(argv);
//  }
//  catch(const std::string& s)
//  {
//    print_usage(argv);
//    return false;
//  }

  // Get image filenames.
  vector<string> filenames;
  getImageFilePaths(filenames, srcPath("../../datasets/"));

  
	return 0;
}