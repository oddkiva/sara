#include <DO/Sara/Core.hpp>
#include <DO/Sara/FileSystem.hpp>

namespace sara = DO::Sara;

const auto dirpath = std::string{"/home/david/Desktop/Datasets/sfm/castle_int"};
auto images = std::vector<std::string>{};
const auto png_files = sara::ls(dirpath, ".png");
append(images, png_files);
//append(images, sara::ls(dirpath, ".jpg"));
