// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/FileSystem/FileSystem.hpp>

#if __has_include(<filesystem>)
#  include <filesystem>
#else
// N.B.: Boost 1.74 `copy_file` function is buggy...
#  include <boost/filesystem.hpp>
#endif

#include <algorithm>
#include <iostream>
#include <stdexcept>


#if __has_include(<filesystem>)
namespace fs = std::filesystem;
#else
namespace fs = boost::filesystem;
#endif


namespace DO::Sara {

  auto basename(const std::string& filepath) -> std::string
  {
#if __has_include(<filesystem>)
    return fs::path{filepath}.stem().string();
#else
    return fs::basename(filepath);
#endif
  }

  auto mkdir(const std::string& dirpath) -> void
  {
    if (fs::is_directory(dirpath))
      return;
    fs::create_directory(dirpath);
  }

  auto cp(const std::string& from, const std::string& to) -> void
  {
#if __has_include(<filesystem>)
    fs::copy_file(from, to, fs::copy_options::overwrite_existing);
#else
    fs::copy_file(from, to, fs::copy_option::overwrite_if_exists);
#endif
  }

  auto ls(const std::string& dirpath, const std::string& ext_filter)
      -> std::vector<std::string>
  {
#if __has_include(<filesystem>)
    auto in_path = fs::absolute(dirpath);
#else
    auto in_path = fs::system_complete(fs::path{dirpath});
#endif

    if (!fs::exists(in_path))
      throw std::runtime_error{"Error: directory does not exist"};

    if (!fs::is_directory(in_path))
      throw std::runtime_error{"Error: " + in_path.string() +
                               " is not a folder!"};

    // Now parsing...
    SARA_DEBUG << "Parsing file directory: "
               << "'" << fs::canonical(in_path).string() << "'" << std::endl;

    auto filepaths = std::vector<std::string>{};

    auto end_iter = fs::directory_iterator{};
    for (auto dir_i = fs::directory_iterator{in_path}; dir_i != end_iter;
         ++dir_i)
    {
      if (!fs::is_regular_file(dir_i->status()))
        continue;

#if __has_include(<filesystem>)
      auto ext = dir_i->path().extension().string();
#else
      auto ext = fs::extension(dir_i->path());
#endif
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

      if (ext == ext_filter)
        filepaths.push_back(fs::canonical(dir_i->path()).string());
    }

    if (filepaths.empty())
      SARA_DEBUG << "Did not found files with extension: " << ext_filter
                 << std::endl;
    else
    {
      SARA_DEBUG << "Found:" << std::endl;
      for (auto i = 0u; i < filepaths.size(); ++i)
        SARA_DEBUG << "[" << i << "]  " << filepaths[i] << std::endl;
    }


    return filepaths;
  }

} /* namespace DO::Sara */
