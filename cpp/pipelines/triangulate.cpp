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

#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace std;
using namespace DO::Sara;


void triangulate(const std::string& dirpath, const std::string& h5_filepath,
                 bool overwrite, bool debug)
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    cp(h5_filepath, h5_filepath + ".bak");

  SARA_DEBUG << "Opening file " << h5_filepath << "..." << std::endl;
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDWR};

  auto view_attributes = ViewAttributes{};

  // Load images (necessary if we want to extract the colors).
  SARA_DEBUG << "Listing images from:\n\t" << dirpath << std::endl;
  view_attributes.list_images(dirpath);
  view_attributes.read_images();

  // Load keypoints.
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  view_attributes.read_keypoints(h5_file);

  // Load the internal camera matrices from Strecha dataset.
  // N.B.: this is an ad-hoc code.
  SARA_DEBUG << "Reading internal camera matrices in Strecha's data format"
             << std::endl;
  std::for_each(
      std::begin(view_attributes.image_paths),
      std::end(view_attributes.image_paths), [&](const auto& image_path) {
        const auto K_filepath = dirpath + "/" + basename(image_path) + ".png.K";
        SARA_DEBUG << "Reading internal camera matrix from:\n\t" << K_filepath
                   << std::endl;
        view_attributes.cameras.push_back(normalized_camera());
        view_attributes.cameras.back().K =
            read_internal_camera_parameters(K_filepath);
      });

  // Initialize the epipolar graph.
  const auto num_vertices = int(view_attributes.image_paths.size());
  SARA_CHECK(num_vertices);

  auto edge_attributes = EpipolarEdgeAttributes{};
  SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
  edge_attributes.initialize_edges(num_vertices);

  SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath << std::endl;
  edge_attributes.read_matches(h5_file, view_attributes);

  SARA_DEBUG << "Reading the essential matrices..." << std::endl;
  edge_attributes.resize_essential_edge_list();
  edge_attributes.read_essential_matrices(view_attributes, h5_file);

  // Convenient references.
  const auto& edge_ids = edge_attributes.edge_ids;
  const auto& edges = edge_attributes.edges;
  const auto& matches = edge_attributes.matches;

  const auto& E = edge_attributes.E;
  const auto& E_num_samples = edge_attributes.E_num_samples;
  const auto& E_noise = edge_attributes.E_noise;
  const auto& E_best_samples = edge_attributes.E_best_samples;
  const auto& E_inliers = edge_attributes.E_inliers;

  int display_step = 20;

  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;
        const auto& Mij = matches[ij];

        const auto& Eij = E[ij];
        const auto& E_inliers_ij = E_inliers[ij];
        const auto& E_best_sample_ij = E_best_samples[ij];

        const auto& Ki = view_attributes.cameras[i].K;
        const auto& Kj = view_attributes.cameras[j].K;
        SARA_DEBUG << "Internal camera matrices :\n"
                   << "- K[" << i << "] =\n" << Ki << "\n"
                   << "- K[" << j << "] =\n" << Kj << "\n";
        const Matrix3d Ki_inv = Ki.inverse();
        const Matrix3d Kj_inv = Kj.inverse();


        // =====================================================================
        // Check the epipolar geometry first.
        //
        if (debug)
        {
          SARA_DEBUG
              << "Forming the fundamental matrix from the essential matrix:\n";
          std::cout.flush();
          auto Fij = FundamentalMatrix{};
          Fij.matrix() = Kj.inverse().transpose() * Eij.matrix() * Ki.inverse();

          SARA_DEBUG << "Fij = \n" << Fij << std::endl;
          SARA_CHECK(E_num_samples[ij]);
          SARA_CHECK(E_noise[ij]);
          SARA_CHECK(E_inliers[ij].row_vector().count());
          SARA_CHECK(E_best_samples[ij].row_vector());

          const auto& Ii = view_attributes.images[i];
          const auto& Ij = view_attributes.images[j];
          check_epipolar_constraints(Ii, Ij, Fij, Mij, E_best_sample_ij,
                                     E_inliers_ij, display_step,
                                     /* wait_key */ false);
        }


        // =====================================================================
        // Perform triangulation.
        //
        print_stage("Performing data transformations...");
        // Tensors of image coordinates.
        const auto& fi = features(view_attributes.keypoints[i]);
        const auto& fj = features(view_attributes.keypoints[j]);
        const auto ui = homogeneous(extract_centers(fi)).template cast<double>();
        const auto uj = homogeneous(extract_centers(fj)).template cast<double>();
        // Tensors of camera coordinates.
        const auto uni = apply_transform(Ki_inv, ui);
        const auto unj = apply_transform(Kj_inv, uj);
        static_assert(std::is_same_v<decltype(uni), const Tensor_<double, 2>>);
        // List the matches as a 2D-tensor where each row encodes a match 'm' as
        // a pair of point indices (i, j).
        const auto index_matches_ij = to_tensor(Mij);

        // Reminder: do not try to calculate the two-view geometry if the
        // essential matrix estimation failed.
        if (E_inliers_ij.flat_array().count() == 0)
          return;

        // Estimate the two-view geometry, i.e.:
        // 1. Find the cheiral-most relative camera motion.
        // 2. Calculate the 3D points from every feature matches
        // 3. Calculate their cheirality.
        auto geometry = estimate_two_view_geometry(index_matches_ij, uni, unj,
                                                   Eij, E_inliers_ij, E_best_sample_ij);

        // I choose to keep every point information inliers or not, cheiral or
        // not.
        //
        // The following function can be used to keep cheiral inliers and view
        // the cleaned point cloud.
        // keep_cheiral_inliers(geometry, inliers);

        // Add the internal camera matrices to the camera.
        geometry.C1.K = Ki;
        geometry.C2.K = Kj;

        const auto colors = extract_colors(view_attributes.images[i],
                                           view_attributes.images[j], geometry);

        // Save the data to HDF5.
        h5_file.get_group("two_view_geometries");
        h5_file.get_group("two_view_geometries/cameras");
        h5_file.get_group("two_view_geometries/points");
        h5_file.get_group("two_view_geometries/cheirality");
        h5_file.get_group("two_view_geometries/colors");
        {
          // Get the left and right cameras.
          auto cameras = Tensor_<PinholeCamera, 1>{2};
          cameras(0) = geometry.C1;
          cameras(1) = geometry.C2;

          const MatrixXd X_euclidean = geometry.X.colwise().hnormalized();
          SARA_DEBUG << "X =\n" << X_euclidean.leftCols(20) << std::endl;
          SARA_DEBUG << "cheirality =\n" << geometry.cheirality.leftCols(20) << std::endl;

          h5_file.write_dataset(
              format("two_view_geometries/cameras/%d_%d", i, j), cameras,
              overwrite);
          h5_file.write_dataset(
              format("two_view_geometries/points/%d_%d", i, j), X_euclidean,
              overwrite);
          h5_file.write_dataset(
              format("two_view_geometries/cheirality/%d_%d", i, j),
              geometry.cheirality, overwrite);
          h5_file.write_dataset(
              format("two_view_geometries/colors/%d_%d", i, j), colors,
              overwrite);
        }
      });
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
        ("debug", "Inspect visually the epipolar geometry")            //
        ("overwrite", "Overwrite triangulation")                       //
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
      std::cout << "[--dirpath]: missing image directory path" << std::endl;
      return 0;
    }
    if (!vm.count("out_h5_file"))
    {
      std::cout << desc << std::endl;
      std::cout << "[--out_h5_file]: missing output H5 file path" << std::endl;
      return 0;
    }

    const auto dirpath = vm["dirpath"].as<std::string>();
    const auto h5_filepath = vm["out_h5_file"].as<std::string>();
    const auto debug = vm.count("debug");
    const auto overwrite = vm.count("overwrite");
    triangulate(dirpath, h5_filepath, overwrite, debug);

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
