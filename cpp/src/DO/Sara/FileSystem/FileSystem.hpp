#pragma once

#include <experimental/filesystem>
#include <string>
#include <vector>


namespace fs = std::experimental::filesystem;


namespace DO::Sara {

auto basename(const std::string& filepath) -> std::string;

auto mkdir(const std::string& dirpath) -> void;

auto cp(const std::string& from, const std::string& to) -> void;

auto ls(const std::string& dirpath, const std::string& ext_filter)
    -> std::vector<std::string>;

} /* namespace DO::Sara */
