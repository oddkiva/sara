#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>


namespace sara = DO::Sara;


int main(int, char**)
{
  const auto dirpath = fs::path{"/Users/david/Desktop/Datasets/sfm/castle_int"};
  auto image_paths = sara::ls(dirpath.string(), ".png");

  auto h5_file = sara::H5File{
      "/Users/david/Desktop/Datasets/sfm/castle_int.h5", H5F_ACC_TRUNC};

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = sara::imread<float>(path);

        SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
        const auto keys = sara::compute_sift_keypoints(image);

        const auto group_name = sara::basename(path);
        h5_file.group(group_name);

        const auto& [f, v] = keys;

        h5_file.write_dataset(group_name + "/" + "features", tensor_view(f));
        h5_file.write_dataset(group_name + "/" + "descriptors", v);
      });

  return 0;
}
