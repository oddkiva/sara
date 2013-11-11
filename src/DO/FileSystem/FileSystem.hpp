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

	bool createDirectory(const std::string& dirName);

	bool copyFile(const std::string& from, const std::string& to);

  std::string parentDirectory(const std::string& filepath);

	std::string basename(const std::string& filepath);

	bool getImageFilePaths(
    std::vector<std::string>& filePaths,
    const std::string& dirName);

  bool getFilePaths(
    std::vector<std::string>& filePaths,
    const std::string& dirName,
    const std::string& nameFilter);

} /* namespace DO */

#endif /* DO_FILESYSTEM_FILESYSTEM_HPP */