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

  std::for_each(std::begin(image_paths), std::end(image_paths),
                [&](const auto& path) {
                  SARA_DEBUG << "Reading image " << path << "..." << std::endl;
                  auto image = sara::imread<float>(path);

                  SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
                  auto keypoints = sara::compute_sift_keypoints(image);

                  auto group_name = sara::basename(path);
                  h5_file.group(group_name);

                  //const auto features_view = sara ::TensorView_<sara::OERegion, 1>{
                  //    keypoints.features.data(), keypoints.features.size()};
                  //path.write_dataset(group_name + "/" + "features",
                  //                   features_view);

                  //const auto descriptors_view = sara::TensorView_<float, 2>{
                  //    keypoints.descriptors.matrix().data(),
                  //    {keypoints.descriptors.dimension(),
                  //     keypoints.descriptors.size()}};
                  //path.write_dataset(group_name + "/" + "descriptors",
                  //                   descriptors_view);
                });

  return 0;
}
