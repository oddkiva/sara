#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/FileSystem/FileSystem.hpp>

#include <boost/filesystem.hpp>

#include <algorithm>
#include <iostream>


namespace fs = boost::filesystem;


namespace DO::Sara {

auto basename(const std::string& filepath) -> std::string
{
  return fs::basename(filepath);
}

auto mkdir(const std::string& dirpath) -> void
{
  if (fs::is_directory(dirpath))
    return;
  fs::create_directory(dirpath);
}

auto cp(const std::string& from, const std::string& to) -> void
{
  fs::copy_file(from, to, fs::copy_option::overwrite_if_exists);
}

auto ls(const std::string& dirpath, const std::string& ext_filter)
    -> std::vector<std::string>
{
  auto in_path = fs::system_complete(fs::path{dirpath});

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
  for (auto dir_i = fs::directory_iterator{in_path}; dir_i != end_iter; ++dir_i)
  {
    if (!fs::is_regular_file(dir_i->status()))
      continue;

    auto ext = fs::extension(dir_i->path());
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ext_filter)
      filepaths.push_back(fs::canonical(dir_i->path()).string());
  }

  if (filepaths.empty())
    SARA_DEBUG << "Did not found files with extension: " << ext_filter << std::endl;
  else
  {
    SARA_DEBUG << "Found:" << std::endl;
    for (auto i = 0u; i < filepaths.size(); ++i)
      SARA_DEBUG << "[" << i << "]  " << filepaths[i] << std::endl;
  }


  return filepaths;
}

} /* namespace DO::Sara */
