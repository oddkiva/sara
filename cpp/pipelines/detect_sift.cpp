#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


namespace sara = DO::Sara;


int main(int, char **)
{
  const auto dirpath = fs::path{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto image_names = sara::ls(dirpath.string(), ".png");

  auto image_paths = std::vector<fs::path>{};
  std::transform(image_names.begin(), image_names.end(),
                 std::back_inserter(image_paths),
                 [&dirpath](const auto& name) { return dirpath / name; });

  std::for_each(std::begin(image_paths), std::end(image_paths),
                [](const auto& path) {
                  auto image = sara::imread<float>(path.string());
                  sara::compute_sift_keypoints(image);
                });

  return 0;
}
