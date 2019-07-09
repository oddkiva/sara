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
#include <DO/Sara/Match.hpp>
#include <DO/Sara/Match/IndexMatch.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/Datasets/Strecha.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;


auto estimate_fundamental_matrix(
    const std::vector<sara::Match>& Mij,
    const sara::KeypointList<sara::OERegion, float>& ki,
    const sara::KeypointList<sara::OERegion, float>& kj, int num_samples,
    double err_thres)
{
  const auto to_double = [](const float& src) { return double(src); };
  const auto& fi = features(ki);
  const auto& fj = features(kj);
  const auto pi = extract_centers(fi).cwise_transform(to_double);
  const auto pj = extract_centers(fj).cwise_transform(to_double);

  const auto Pi = homogeneous(pi);
  const auto Pj = homogeneous(pj);

  const auto Mij_tensor = to_tensor(Mij);

  auto estimator = sara::EightPointAlgorithm{};
  auto distance = sara::EpipolarDistance{};

  const auto [F, num_inliers, sample_best] = sara::ransac(
      Mij_tensor, Pi, Pj, estimator, distance, num_samples, err_thres);

  SARA_CHECK(F);
  SARA_CHECK(num_inliers);
  SARA_CHECK(Mij.size());

  return std::make_tuple(F, num_inliers, sample_best);
}

auto check_epipolar_constraints(const sara::Image<sara::Rgb8>& Ii,
                                const sara::Image<sara::Rgb8>& Ij,
                                const sara::FundamentalMatrix& F,
                                const std::vector<sara::Match>& Mij,
                                const sara::Tensor_<int, 1>& sample_best,
                                double err_thres, int display_step)
{
  const auto scale = 0.25f;
  const auto w = int((Ii.width() + Ij.width()) * scale + 0.5f);
  const auto h = int(max(Ii.height(), Ij.height()) * scale + 0.5f);

  if (!sara::active_window())
  {
    sara::create_window(w, h);
    sara::set_antialiasing();
  }

  if (sara::get_sizes(sara::active_window()) != Eigen::Vector2i(w, h))
    sara::resize_window(w, h);

  sara::PairWiseDrawer drawer(Ii, Ij);
  drawer.set_viz_params(scale, scale, sara::PairWiseDrawer::CatH);

  drawer.display_images();

  auto distance = sara::EpipolarDistance{F.matrix()};

  for (size_t m = 0; m < Mij.size(); ++m)
  {
    const Eigen::Vector3d X1 = Mij[m].x_pos().cast<double>().homogeneous();
    const Eigen::Vector3d X2 = Mij[m].y_pos().cast<double>().homogeneous();

    if (distance(X1, X2) > err_thres)
      continue;

    if (m % display_step == 0)
    {
      drawer.draw_match(Mij[m], sara::Blue8, false);

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), sara::Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), sara::Cyan8, 1);
    }
  }

  for (size_t m = 0; m < sample_best.size(); ++m)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(Mij[sample_best(m)], sara::Red8, true);

    const Eigen::Vector3d X1 =
        Mij[sample_best(m)].x_pos().cast<double>().homogeneous();
    const Eigen::Vector3d X2 =
        Mij[sample_best(m)].y_pos().cast<double>().homogeneous();

    const auto proj_X1 = F.right_epipolar_line(X1);
    const auto proj_X2 = F.left_epipolar_line(X2);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), sara::Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), sara::Magenta8, 1);
  }

  //get_key();
}

void estimate_fundamental_matrices(const std::string& dirpath, const std::string& h5_filepath)
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    sara::cp(h5_filepath, h5_filepath + ".bak");

  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDWR};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, sara::ls(dirpath, ".png"));
  append(image_paths, sara::ls(dirpath, ".jpg"));
  std::sort(image_paths.begin(), image_paths.end());

  auto group_names = std::vector<std::string>{};
  group_names.reserve(image_paths.size());
  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::back_inserter(group_names),
                 [&](const std::string& image_path) {
                   return sara::basename(image_path);
                 });

  auto keypoints = std::vector<sara::KeypointList<sara::OERegion, float>>{};
  keypoints.reserve(image_paths.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(keypoints),
                 [&](const std::string& group_name) {
                   return sara::read_keypoints(h5_file, group_name);
                 });

  const auto N = int(image_paths.size());
  auto edges = std::vector<sara::EpipolarEdge>{};
  edges.reserve(N * (N - 1) / 2);
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      edges.push_back({i, j, Eigen::Matrix3d::Zero()});

  auto index_matches = std::vector<std::vector<sara::IndexMatch>>{};
  index_matches.reserve(edges.size());
  std::transform(std::begin(edges), std::end(edges),
                 std::back_inserter(index_matches),
                 [&](const sara::EpipolarEdge& edge) {
                   const auto i = edge.i;
                   const auto j = edge.j;

                   const auto match_dataset = std::string{"matches"} + "/" +
                                              std::to_string(i) + "_" +
                                              std::to_string(j);

                   auto mij = std::vector<sara::IndexMatch>{};
                   h5_file.read_dataset(match_dataset, mij);

                   return mij;
                 });

  const auto edge_ids = sara::range(edges.size());

  auto matches = std::vector<std::vector<sara::Match>>{};
  matches.reserve(edges.size());
  std::transform(std::begin(edge_ids), std::end(edge_ids),
                 std::back_inserter(matches), [&](int ij) {
                 const auto i = edges[ij].i;
                 const auto j = edges[ij].j;
                   return sara::to_match(index_matches[ij], keypoints[i], keypoints[j]);
                 });

  const auto num_samples = 1000;
  const auto f_err_thres = 5e-3;
  auto& f_edges = edges;
  std::transform(
      std::begin(edge_ids), std::end(edge_ids), std::begin(f_edges),
      [&](const auto& edge_id) -> sara::EpipolarEdge {
        const auto& edge = f_edges[edge_id];
        const auto i = edge.i;
        const auto j = edge.j;
        const auto& Mij = matches[edge_id];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        // Estimate the fundamental matrix.
        const auto [F, num_inliers, sample_best] =
            estimate_fundamental_matrix(Mij, ki, kj, num_samples, f_err_thres);

        // Debug.
        const int display_step = 20;
        const auto Ii = sara::imread<sara::Rgb8>(image_paths[i]);
        const auto Ij = sara::imread<sara::Rgb8>(image_paths[j]);
        check_epipolar_constraints(Ii, Ij, F, Mij, sample_best, f_err_thres,
                                   display_step);

        return num_inliers < 100
                   ? sara::EpipolarEdge{i, j, Eigen::Matrix3d::Zero()}
                   : sara::EpipolarEdge{i, j, F.matrix()};
      });

  // Save F-edges.
  h5_file.write_dataset("f_edges", tensor_view(f_edges));
}


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Estimate fundamental matrices"};
    desc.add_options()                                                 //
        ("help, h", "Help screen")                                     //
        ("dirpath", po::value<std::string>(), "Image directory path")  //
        ("out_h5_file", po::value<std::string>(), "Output HDF5 file")  //
        ("read", "Visualize detected keypoints")  //
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      return 0;
    }

    if (!vm.count("dirpath"))
    {
      std::cout << "Missing image directory path" << std::endl;
      return 0;
    }
    if (!vm.count("out_h5_file"))
    {
      std::cout << desc << std::endl;
      std::cout << "Missing output H5 file path" << std::endl;
      return 0;
    }

    const auto dirpath = vm["dirpath"].as<std::string>();
    const auto h5_filepath = vm["out_h5_file"].as<std::string>();
    estimate_fundamental_matrices(dirpath, h5_filepath);

    return 0;
  }
  catch (const po::error& e)
  {
    std::cerr << e.what() << "\n";
    return 1;
  }
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
