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
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/MultiViewGeometry/Datasets/Strecha.hpp>
#include <DO/Sara/MultiViewGeometry/EpipolarGraph.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;

using ESolver = sara::NisterFivePointAlgorithm;


auto estimate_essential_matrix(
    const std::vector<sara::Match>& Mij,
    const sara::KeypointList<sara::OERegion, float>& ki,
    const sara::KeypointList<sara::OERegion, float>& kj,
    const Eigen::Matrix3d& Ki_inv, const Eigen::Matrix3d& Kj_inv,
    int num_samples, double err_thres)
{
  const auto to_double = [](const float& src) { return double(src); };
  const auto& fi = features(ki);
  const auto& fj = features(kj);
  const auto pi = extract_centers(fi).cwise_transform(to_double);
  const auto pj = extract_centers(fj).cwise_transform(to_double);

  const auto Pi = apply_transform(Ki_inv, homogeneous(pi));
  const auto Pj = apply_transform(Kj_inv, homogeneous(pj));

  const auto Mij_tensor = to_tensor(Mij);

  auto estimator = ESolver{};
  auto distance = sara::EpipolarDistance{};

  const auto [E, inliers, sample_best] =
      ransac(Mij_tensor, Pi, Pj, estimator, distance, num_samples, err_thres);

  SARA_CHECK(E);
  SARA_CHECK(num_inliers);
  SARA_CHECK(Mij.size());

  return std::make_tuple(E, inliers, sample_best);
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
}

void estimate_essential_matrices(const std::string& dirpath,
                                 const std::string& h5_filepath, bool overwrite)
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    sara::cp(h5_filepath, h5_filepath + ".bak");

  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDWR};

  auto photo_attributes = sara::PhotoAttributes{};
  photo_attributes.list_images(dirpath);
  photo_attributes.read_keypoints(h5_file);
  const auto& image_paths = photo_attributes.image_paths;
  const auto& group_names = photo_attributes.group_names;
  const auto& keypoints = photo_attributes.keypoints;

  auto K_invs = std::vector<Eigen::Matrix3d>{};
  K_invs.reserve(group_names.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(K_invs),
                 [&](const std::string& group_name) {
                   return sara::read_internal_camera_parameters(
                              dirpath + "/" + group_name + ".png.K")
                       .inverse();
                 });

  const auto num_photos = int(photo_attributes.image_paths.size());

  auto f_edge_attributes = sara::EpipolarEdgeAttributes{};
  f_edge_attributes.initialize_edges(num_photos);
  f_edge_attributes.read_matches(h5_file, photo_attributes);

  auto& f_edges = f_edge_attributes.f_edges;
  h5_file.read_dataset("f_edges", f_edges);

  auto e_edge_attributes = f_edge_attributes;

  const auto& matches = e_edge_attributes.matches;

  auto& e_edge_ids = e_edge_attributes.edge_ids;
  auto& e_edges = e_edge_attributes.e_edges;
  auto& e_noise = e_edge_attributes.e_noise;
  auto& e_num_inliers = e_edge_attributes.e_inliers;
  auto& e_best_samples = e_edge_attributes.e_best_samples;

  // Preallocate space.
  e_noise.resize(e_edge_ids.size());
  e_num_inliers.resize(e_edge_ids.size());
  e_best_samples.resize(int(e_edge_ids.size()), ESolver::num_points);

  const auto num_samples = 1000;
  const auto e_err_thres = 5e-3;
  std::for_each(
      std::begin(e_edge_ids), std::end(e_edge_ids),
      [&](const auto& edge_id) {
        auto& e_edge = e_edges[edge_id];
        const auto i = e_edge.i;
        const auto j = e_edge.j;
        const auto& Mij = matches[edge_id];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        const auto& Ki_inv = K_invs[i];
        const auto& Kj_inv = K_invs[j];

        const auto [E, num_inliers, best_sample] = estimate_essential_matrix(
            Mij, ki, kj, Ki_inv, Kj_inv, num_samples, e_err_thres);

        const Eigen::Matrix3d F = Kj_inv.transpose() * E.matrix() * Ki_inv;

        // Debug.
        const int display_step = 20;
        const auto Ii = sara::imread<sara::Rgb8>(image_paths[i]);
        const auto Ij = sara::imread<sara::Rgb8>(image_paths[j]);
        check_epipolar_constraints(Ii, Ij, F, Mij, best_sample, e_err_thres,
                                   display_step);

        // Update.
        e_edges[edge_id] = sara::EpipolarEdge{i, j, E.matrix()};
        e_num_inliers[edge_id] = num_inliers;
        e_best_samples[edge_id].flat_array() = best_sample.flat_array();
        e_noise[edge_id] = e_err_thres;
      });

  // Save E-edges.
  h5_file.write_dataset("e_edges", sara::tensor_view(e_edges), overwrite);
  h5_file.write_dataset("e_num_inliers", sara::tensor_view(e_num_inliers),
                        overwrite);
  h5_file.write_dataset("e_best_samples", e_best_samples, overwrite);
  h5_file.write_dataset("e_noise", sara::tensor_view(e_noise), overwrite);
}


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Estimate essential matrices"};
    desc.add_options()                                                 //
        ("help, h", "Help screen")                                     //
        ("dirpath", po::value<std::string>(), "Image directory path")  //
        ("out_h5_file", po::value<std::string>(), "Output HDF5 file")  //
        ("overwrite", "Overwrite essential matrices and metadata")     //
        ("read", "Visualize detected keypoints")                       //
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
    const auto overwrite = vm.count("overwrite");
    estimate_essential_matrices(dirpath, h5_filepath, overwrite);

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
