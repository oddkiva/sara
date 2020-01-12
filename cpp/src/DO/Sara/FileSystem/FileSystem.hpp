#pragma once

#include <DO/Sara/Defines.hpp>

#include <string>
#include <vector>


namespace DO::Sara {

  /*!
   *  @ingroup FileSystem
   *  @brief File system utility functions.
   *
   *  @{
   */

  DO_SARA_EXPORT
  auto basename(const std::string& filepath) -> std::string;

  DO_SARA_EXPORT
  auto mkdir(const std::string& dirpath) -> void;

  DO_SARA_EXPORT
  auto cp(const std::string& from, const std::string& to) -> void;

  DO_SARA_EXPORT
  auto ls(const std::string& dirpath, const std::string& ext_filter)
      -> std::vector<std::string>;

  //! @}

} /* namespace DO::Sara */
