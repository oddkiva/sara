#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


namespace sara = DO::Sara;


int main(int, char**)
{
  const auto dirpath = fs::path{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto image_paths = sara::ls(dirpath.string(), ".png");

  std::for_each(std::begin(image_paths), std::end(image_paths),
                [](const auto& path) {
                  SARA_DEBUG << "Reading image " << path << "..." << std::endl;
                  auto image = sara::imread<float>(path);

                  SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
                  sara::compute_sift_keypoints(image);
                });

  return 0;
}
