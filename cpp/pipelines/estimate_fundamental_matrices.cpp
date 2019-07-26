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
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>

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

void estimate_fundamental_matrices(const std::string& dirpath,
                                   const std::string& h5_filepath,
                                   bool overwrite)
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    sara::cp(h5_filepath, h5_filepath + ".bak");

  SARA_DEBUG << "Opening file " << h5_filepath << "..." << std::endl;
  auto h5_file = sara::H5File{h5_filepath, H5F_ACC_RDWR};

  auto photo_attributes = sara::PhotoAttributes{};

  // Load images (optional).
  SARA_DEBUG << "Listing images from:\n\t" << dirpath << std::endl;
  photo_attributes.list_images(dirpath);

  // Load keypoints (optional).
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  photo_attributes.read_keypoints(h5_file);
  const auto& image_paths = photo_attributes.image_paths;
  const auto& keypoints = photo_attributes.keypoints;

  // Initialize the epipolar graph.
  const auto num_vertices = int(photo_attributes.image_paths.size());
  SARA_CHECK(num_vertices);

  auto edge_attributes = sara::EpipolarEdgeAttributes{};
  SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
  edge_attributes.initialize_edges(num_vertices);

  SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath << std::endl;
  edge_attributes.read_matches(h5_file, photo_attributes);

  SARA_DEBUG << "Preallocate the F data structures..." << std::endl;
  edge_attributes.resize_fundamental_edge_list();

  const auto& edge_ids = edge_attributes.edge_ids;
  const auto& edges = edge_attributes.edges;
  const auto& matches = edge_attributes.matches;

  // Mutate these.
  auto& F = edge_attributes.F;
  auto& F_num_samples = edge_attributes.F_best_samples;
  auto& F_noise = edge_attributes.F_noise;
  auto& F_inliers = edge_attributes.F_inliers;
  auto& F_best_samples = edge_attributes.F_best_samples;

  // Preallocate space.
  SARA_CHECK(edge_ids.size());
  SARA_CHECK(edges.size());

  const auto num_samples = 1000;
  const auto f_err_thres = 5e-3;
  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& e) {
        const auto& ij = edges[e];
        const auto i = ij.first;
        const auto j = ij.second;
        const auto& Mij = matches[e];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        SARA_DEBUG << "Calculating fundamental matrices between images:\n"
                   << "- image[" << i
                   << "] = " << photo_attributes.group_names[i] << "\n"
                   << "- image[" << j
                   << "] = " << photo_attributes.group_names[j] << "\n";
        std::cout.flush();

        // Estimate the fundamental matrix.
        const auto [Fij, F_inliers_ij, F_best_sample_ij] =
            estimate_fundamental_matrix(Mij, ki, kj, num_samples, f_err_thres);
        SARA_DEBUG << "F = \n" << F << std::endl;
        SARA_CHECK(num_inliers);
        SARA_CHECK(best_sample.row_vector());

        // Debug.
        const int display_step = 20;
        const auto Ii = sara::imread<sara::Rgb8>(image_paths[i]);
        const auto Ij = sara::imread<sara::Rgb8>(image_paths[j]);
        check_epipolar_constraints(Ii, Ij, Fij, Mij, F_best_sample_ij, f_err_thres,
                                   display_step);

        // Update.
        F[ij] = Fij;
        F_inliers[ij] = F_inliers_ij;
        F_best_samples[ij] = F_best_sample_ij;
        F_noise[ij] = f_err_thres;
      });

  h5_file.write_dataset("F", sara::tensor_view(F), overwrite);
  h5_file.write_dataset("F_num_samples", sara::tensor_view(F_num_samples), overwrite);
  h5_file.write_dataset("F_noise", sara::tensor_view(F_noise), overwrite);
  h5_file.write_dataset("F_best_samples", F_best_samples, overwrite);
  // Save F-edges.
  std::for_each(std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
                const auto i = edges[ij].first;
                const auto j = edges[ij].second;
                h5_file.write_dataset(sara::format("F_inliers/%i/%j", i, j),
                                      sara::tensor_view(F_inliers[ij]),
                                      overwrite);
  });
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
        ("overwrite", "Overwrite fundamental matrices and meta data")  //
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
    const auto overwrite = vm.count("overwrite");

    estimate_fundamental_matrices(dirpath, h5_filepath, overwrite);

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
