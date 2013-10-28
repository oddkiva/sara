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
#include <boost/filesystem.hpp>
#include <algorithm>
#include <cctype>
#include <iostream>

using namespace std;
namespace fs = boost::filesystem;

namespace DO {	

	typedef pair<size_t, size_t> Match;

	bool createDirectory(const string& dirName)
	{
		if(fs::is_directory(dirName))
			return true;
		try
		{
			fs::create_directory(dirName);
		}
		catch(const std::exception & e)
		{ 
			cout << e.what() << endl;
			return false;
		}
		return true;
	}

	bool copyFile(const string& from, const string& to)
	{
		try
		{ 
			fs::copy_file(
				from, to, 
				fs::copy_option::overwrite_if_exists
				);
		}
		catch (const std::exception & e)
		{ 
			cout << e.what() << endl;
			return false;
		}
		return true;
	}

  string parentDirectory(const string& filepath)
  {
    return fs::path(filepath).parent_path().string();
  }

	string basename(const string& filepath)
	{
    return fs::basename(filepath);
	}

	vector<string> populateKeyFileNames(const vector<string>& imageFileNames)
	{
		vector<string> keyFileNames;
		for(vector<string>::const_iterator img = imageFileNames.begin();
			img < imageFileNames.end(); ++img)
			keyFileNames.push_back(basename(*img)+".affkey");
		return keyFileNames;
	}

	bool getImageFilePaths(vector<string> & filePaths, const string& dirName)
	{
		// Retrieve the full path of the directory.
		fs::path inPath(fs::system_complete(fs::path(dirName)));

		// Does the path exist?
		if(!fs::exists(inPath))
		{
			cerr << "\nError: cannot find directory ";
			cerr << inPath.string() << endl;
			return false;
		}

		// Is it a directory?
		if(!fs::is_directory(inPath))
		{
			cerr << "\nError: " << inPath.string();
			cerr << " is not a folder!" << endl;
			return false;
		}

		// Now parsing...
		cout << "\nParsing image directory: '" << inPath.string() << "'" << endl;
		cout << "Found:" << endl;

		size_t imageCount = 0;
		fs::directory_iterator end_iter;
		for(fs::directory_iterator dirIt(inPath);
			dirIt != end_iter; ++dirIt, ++imageCount)
		{
			try
			{
				if(!fs::is_regular_file(dirIt->status()))
					continue;
				string ext = fs::extension(dirIt->path());
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
				if( ext == ".jpg" || ext == ".tif" || ext == ".jpeg" ||
					ext == ".png" || ext == ".ppm" )
				{
					cout << "[" << imageCount << "]\t'" << dirIt->path().filename() << "'" << endl;
					filePaths.push_back(dirIt->path().string());
				}
			}
			catch(const std::exception& e)
			{
				cout << dirIt->path().filename() << " ";
				cout << e.what() << endl;
			}
		}
		return true;
	}

  bool getFilePaths(vector<string>& filePaths, const string& dirName, const string& nameFilter)
  {
    // Retrieve the full path of the directory.
    fs::path inPath(fs::system_complete(fs::path(dirName)));

    // Does the path exist?
    if(!fs::exists(inPath))
    {
      cerr << "\nError: cannot find directory ";
      cerr << inPath.string() << endl;
      return false;
    }

    // Is it a directory?
    if(!fs::is_directory(inPath))
    {
      cerr << "\nError: " << inPath.string();
      cerr << " is not a folder!" << endl;
      return false;
    }

    // Now parsing...
    cout << "\nParsing file directory: '" << inPath.string() << "'" << endl;
    cout << "Found:" << endl;

    size_t fileCount = 0;
    fs::directory_iterator end_iter;
    for(fs::directory_iterator dirIt(inPath); dirIt != end_iter; ++dirIt)
    {
      try
      {
        if(!fs::is_regular_file(dirIt->status()))
          continue;
        string ext = fs::extension(dirIt->path());
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if( ext == nameFilter )
        {
          cout << "[" << fileCount << "]\t'" << dirIt->path().filename() << "'" << endl;
          filePaths.push_back(dirIt->path().string());
          ++fileCount;
        }
      }
      catch(const std::exception& e)
      {
        cout << dirIt->path().filename() << " ";
        cout << e.what() << endl;
      }
    }
    return true;
  }

} /* namespace DO */