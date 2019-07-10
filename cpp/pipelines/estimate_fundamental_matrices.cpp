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


using FSolver = sara::EightPointAlgorithm;


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

  auto estimator = FSolver{};
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

  auto pose_attributes = sara::PoseAttributes{};
  pose_attributes.list_images(dirpath);
  pose_attributes.load_keypoints(h5_file);

  const auto num_vertices = int(pose_attributes.image_paths.size());

  auto f_edge_attributes = sara::EpipolarEdgeAttributes{};
  f_edge_attributes.initialize_edges(num_vertices);
  f_edge_attributes.read_matches(h5_file, pose_attributes);

  const auto& f_edge_ids = f_edge_attributes.edge_ids;

  // Mutate these.
  auto& f_edges = f_edge_attributes.edges;
  auto& f_noise = f_edge_attributes.noise;
  auto& f_num_inliers = f_edge_attributes.num_inliers;
  auto& f_best_samples = f_edge_attributes.best_samples;

  // Preallocate space.
  num_inliers.resize(f_edge_ids.size());
  best_sample.resize(int(f_edge_ids.size()), FSolver::num_points);

  const auto num_samples = 1000;
  const auto f_err_thres = 5e-3;
  std::for_each(
      std::begin(f_edge_ids), std::end(f_edge_ids),
      [&](const auto& edge_id) -> sara::EpipolarEdge {
        const auto& edge = f_edges[edge_id];
        const auto i = edge.i;
        const auto j = edge.j;
        const auto& Mij = matches[edge_id];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        // Estimate the fundamental matrix.
        const auto [F, num_inliers, best_sample] =
            estimate_fundamental_matrix(Mij, ki, kj, num_samples, f_err_thres);

        // Debug.
        const int display_step = 20;
        const auto Ii = imread<sara::Rgb8>(image_paths[i]);
        const auto Ij = imread<sara::Rgb8>(image_paths[j]);
        check_epipolar_constraints(Ii, Ij, F, Mij, sample_best, f_err_thres,
                                   display_step);

        // Update.
        f_edges[edge_id] = sara::EpipolarEdge{i, j, F.matrix()};
        f_num_inliers[edge_id] = num_inliers;
        f_best_samples[edge_id].flat_array() = best_sample.flat_array();
        f_noise[edge_id] = f_err_thres;
      });

  // Save F-f_edges.
  h5_file.write_dataset("f_edges", tensor_view(f_edges));
  h5_file.write_dataset("f_num_inliers", tensor_view(f_num_inliers));
  h5_file.write_dataset("f_best_samples", tensor_view(f_best_samples));
  h5_file.write_dataset("f_noise", tensor_view(f_noise));
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
