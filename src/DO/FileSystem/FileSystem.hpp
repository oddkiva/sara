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

#ifndef DO_FILESYSTEM_FILESYSTEM_HPP
#define DO_FILESYSTEM_FILESYSTEM_HPP

#include <utility>
#include <string>
#include <vector>

namespace DO {

	typedef std::vector<std::pair<size_t, size_t> > IndexMatches;

	bool createFolder(const std::string& folder);

	bool copyFile(const std::string& from, const std::string& to);

  std::string getParentFolder(const std::string& filepath);

	std::string getBasename(const std::string& filepath);

  bool getFilePaths(std::vector<std::string>& filePaths,
                    const std::string& folder,
                    const std::string& wildcard,
                    bool verbose = true);
  
  bool getImageFilePaths(std::vector<std::string>& filePaths,
                         const std::string& folder,
                         bool verbose = true);

} /* namespace DO */

#endif /* DO_FILESYSTEM_FILESYSTEM_HPP */