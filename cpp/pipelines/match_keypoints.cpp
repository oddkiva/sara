// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;


namespace DO::Sara
{

auto match(const sara::KeypointList<sara::OERegion, float>& keys1,
           const sara::KeypointList<sara::OERegion, float>& keys2)
    -> std::vector<sara::Match>
{
  sara::AnnMatcher matcher{keys1, keys2, 1.0f};
  return matcher.compute_matches();
}


struct IndexMatch
{
  int i;
  int j;
  float score;
};

struct EpipolarEdge
{
  int i;  // left
  int j;  // right
  Matrix3d m;
};

template <>
struct CalculateH5Type<IndexMatch>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(IndexMatch)};
    INSERT_MEMBER(h5_comp_type, IndexMatch, i);
    INSERT_MEMBER(h5_comp_type, IndexMatch, j);
    INSERT_MEMBER(h5_comp_type, IndexMatch, score);
    return h5_comp_type;
  }
};

template <>
struct CalculateH5Type<EpipolarEdge>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(EpipolarEdge)};
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, i);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, j);
    INSERT_MEMBER(h5_comp_type, EpipolarEdge, m);
    return h5_comp_type;
  }
};


KeypointList<OERegion, float> read_keypoints(H5File& h5_file,
                                             const std::string& group_name)
{
  auto features = std::vector<sara::OERegion>{};
  auto descriptors = sara::Tensor_<float, 2>{};

  SARA_DEBUG << "Read features..." << std::endl;
  h5_file.read_dataset(group_name + "/" + "features", features);

  SARA_DEBUG << "Read descriptors..." << std::endl;
  h5_file.read_dataset(group_name + "/" + "descriptors", descriptors);

  return {features, descriptors};
}

auto read_matches(H5File& file, const std::string& name)
{
  auto matches = std::vector<IndexMatch>{};
  file.read_dataset(name, matches);
  return matches;
}

}  // namespace DO::Sara


GRAPHICS_MAIN()
{
#if defined(__APPLE__)
  const auto dirpath = fs::path{"/Users/david/Desktop/Datasets/sfm/castle_int"};
  auto h5_file = sara::H5File{"/Users/david/Desktop/Datasets/sfm/castle_int.h5",
                              H5F_ACC_RDWR};
#else
  const auto dirpath = fs::path{"/home/david/Desktop/Datasets/sfm/castle_int"};
  auto h5_file = sara::H5File{"/home/david/Desktop/Datasets/sfm/castle_int.h5",
                              H5F_ACC_RDWR};
#endif

  auto image_paths = sara::ls(dirpath.string(), ".png");
  std::sort(image_paths.begin(), image_paths.end());

  const auto N = int(image_paths.size());
  for (int i = 0; i < N; ++i)
  {
    for (int j = i + 1; j < N; ++j)
    {
      const auto& fi = image_paths[i];
      const auto& fj = image_paths[j];

      const auto gi = sara::basename(fi);
      const auto gj = sara::basename(fj);

      SARA_DEBUG << gi << std::endl;
      SARA_DEBUG << gj << std::endl;

      // Load images.
      const auto Ki = sara::read_keypoints(h5_file, gi);
      const auto Kj = sara::read_keypoints(h5_file, gj);

      const auto Mij = match(Ki, Kj);

      auto Mij2 = std::vector<sara::IndexMatch>{};
      std::transform(
          Mij.begin(), Mij.end(), std::back_inserter(Mij2), [](const auto& m) {
            return sara::IndexMatch{m.x_index(), m.y_index(), m.score()};
          });

      const auto group_name = std::string{"matches"};
      h5_file.group(group_name);

      const auto match_dataset =
          group_name + "/" + std::to_string(i) + "_" + std::to_string(j);
      h5_file.write_dataset(match_dataset, tensor_view(Mij2));

      // auto Fij = sara::estimate_fundamental_matrix(Mij);
      // auto Eij = sara::estimate_essential_matrix(Mij);

      const auto Ii = sara::imread<sara::Rgb8>(fi);
      const auto Ij = sara::imread<sara::Rgb8>(fj);

      const auto scale = 0.25f;
      const auto w = int((Ii.width() + Ij.width()) * scale + 0.5f);
      const auto h = int(max(Ii.height(), Ij.height()) * scale + 0.5f);
      const auto off = sara::Point2f{float(Ii.width()), 0.f};

      if (!sara::active_window())
      {
        sara::create_window(w, h);
        sara::set_antialiasing();
      }

      if (sara::get_sizes(sara::active_window()) != Eigen::Vector2i(w, h))
        sara::resize_window(w, h);

      sara::draw_image_pair(Ii, Ij, off, scale);
      for (size_t m = 0; m < 500; ++m)
      {
        sara::draw_match(Mij[m], sara::Red8, off, scale);
        cout << Mij[m] << endl;
        if (m % 100 == 0)
          sara::get_key();
      }
      sara::get_key();
    }
  }

  return 0;
}
