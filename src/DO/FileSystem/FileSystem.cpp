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
#include <stlplus/portability/file_system.hpp>
#include <algorithm>
#include <cctype>
#include <iostream>

using namespace std;

namespace DO {	

	typedef pair<size_t, size_t> Match;

	bool createFolder(const string& dirName)
	{
		if (stlplus::folder_exists(dirName))
			return true;
    return stlplus::folder_create(dirName);
	}

	bool copyFile(const string& from, const string& to)
	{
    return stlplus::file_copy(from, to);
	}

  string getParentFolder(const string& filepath)
  {
    return stlplus::folder_part(filepath);
  }

	string getBasename(const string& filepath)
	{
    return stlplus::basename_part(filepath);
	}


  bool getFilePaths(vector<string>& filePaths, const string& folder,
                    const string& wildcard, bool verbose)
  {
    // Does the path exist?
    if(!stlplus::folder_exists(folder))
    {
      cerr << "\nError: cannot find folder ";
      cerr << folder << endl;
      return false;
    }

    filePaths = stlplus::folder_wildcard(folder, wildcard);

    // Now listing files in folder.
    if (verbose)
    {
      cout << "\nListing files in folder: '" << folder << "'" << endl;
      cout << "Found:" << endl;
      for(size_t i = 0; i != filePaths.size(); ++i)
        cout << "[" << i << "]\t'" << filePaths[i] << "'" << endl;
    }
    return true;
  }
  
  bool getImageFilePaths(vector<string>& filePaths, const string& folder,
                         bool verbose)
  {
    if (!filePaths.empty())
      filePaths.clear();
    
    const string wildcards[7] = {
      "*.jpeg", "*.jpg", "*.jfif", "*.jpe",
      "*.tif", "*.tiff",
      "*.png"
    };
    
    for (int i = 0; i < 7; ++i)
    {
      vector<string> newFilePaths;
      if (!getFilePaths(newFilePaths, folder, wildcards[i], false))
        return false;
      filePaths.insert(filePaths.end(),
                       newFilePaths.begin(), newFilePaths.end());
    }
    
    // Now listing files in folder.
    if (verbose)
    {
      cout << "\nListing files in folder: '" << folder << "'" << endl;
      cout << "Found:" << endl;
      for(size_t i = 0; i != filePaths.size(); ++i)
        cout << "[" << i << "]\t'" << filePaths[i] << "'" << endl;
    }
    return true;
  }
  
  

} /* namespace DO */