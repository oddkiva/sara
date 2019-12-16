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

#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/SfM/BuildingBlocks/EssentialMatrixEstimation.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>

#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;


namespace DO::Sara {

  using ESolver = NisterFivePointAlgorithm;

  auto estimate_essential_matrix(const std::vector<Match>& Mij,            //
                                 const KeypointList<OERegion, float>& ki,  //
                                 const KeypointList<OERegion, float>& kj,  //
                                 const Eigen::Matrix3d& Ki_inv,            //
                                 const Eigen::Matrix3d& Kj_inv,            //
                                 int num_samples,                          //
                                 double err_thres)
      -> std::tuple<EssentialMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>
  {
    const auto& fi = features(ki);
    const auto& fj = features(kj);
    const auto ui = extract_centers(fi).cast<double>();
    const auto uj = extract_centers(fj).cast<double>();

    const auto uni = apply_transform(Ki_inv, homogeneous(ui));
    const auto unj = apply_transform(Kj_inv, homogeneous(uj));

    const auto Mij_tensor = to_tensor(Mij);

    auto estimator = ESolver{};
    auto distance = EpipolarDistance{};

    const auto [E, inliers, sample_best] = ransac(
        Mij_tensor, uni, unj, estimator, distance, num_samples, err_thres);

    SARA_CHECK(E);
    SARA_CHECK(inliers.row_vector());
    SARA_CHECK(Mij.size());

    return std::make_tuple(E, inliers, sample_best);
  }

  auto estimate_essential_matrices(const std::string& dirpath,      //
                                   const std::string& h5_filepath,  //
                                   int num_samples,                 //
                                   double noise,                    //
                                   int min_F_inliers,               //
                                   bool overwrite,                  //
                                   bool debug) -> void
  {
    // Create a backup.
    if (!fs::exists(h5_filepath + ".bak"))
      cp(h5_filepath, h5_filepath + ".bak");

    SARA_DEBUG << "Opening file " << h5_filepath << "..." << std::endl;
    auto h5_file = H5File{h5_filepath, H5F_ACC_RDWR};

    auto view_attributes = ViewAttributes{};

    // Load images (optional).
    SARA_DEBUG << "Listing images from:\n\t" << dirpath << std::endl;
    view_attributes.list_images(dirpath);
    if (debug)
      view_attributes.read_images();

    // Load the internal camera matrices from Strecha dataset.
    // N.B.: this is an ad-hoc code.
    SARA_DEBUG << "Reading internal camera matrices in Strecha's data format"
               << std::endl;
    std::for_each(std::begin(view_attributes.image_paths),
                  std::end(view_attributes.image_paths),
                  [&](const auto& image_path) {
                    const auto K_filepath =
                        dirpath + "/" + basename(image_path) + ".png.K";
                    SARA_DEBUG << "Reading internal camera matrix from:\n\t"
                               << K_filepath << std::endl;
                    view_attributes.cameras.push_back(normalized_camera());
                    view_attributes.cameras.back().K =
                        read_internal_camera_parameters(K_filepath);
                  });

    // Load keypoints.
    SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath
               << std::endl;
    view_attributes.read_keypoints(h5_file);
    const auto& keypoints = view_attributes.keypoints;

    // Initialize the epipolar graph.
    const auto num_vertices = int(view_attributes.image_paths.size());
    SARA_CHECK(num_vertices);

    auto edge_attributes = EpipolarEdgeAttributes{};
    SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
    edge_attributes.initialize_edges(num_vertices);

    SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath
               << std::endl;
    edge_attributes.read_matches(h5_file, view_attributes);

    SARA_DEBUG << "Reading the fundamental matrices..." << std::endl;
    edge_attributes.resize_fundamental_edge_list();
    edge_attributes.read_fundamental_matrices(view_attributes, h5_file);
    // TODO: we will use the meta data later to decide if we want to estimate an
    // essential matrix because it is a lot slower than the fundamental
    // matrix estimation.

    SARA_DEBUG << "Preallocate the E data structures..." << std::endl;
    edge_attributes.resize_essential_edge_list();

    const auto& edge_ids = edge_attributes.edge_ids;
    const auto& edges = edge_attributes.edges;
    const auto& matches = edge_attributes.matches;
    SARA_CHECK(edge_ids.size());
    SARA_CHECK(edges.size());
    SARA_CHECK(matches.size());

    // Mutate these.
    auto& E = edge_attributes.E;
    auto& E_num_samples = edge_attributes.E_num_samples;
    auto& E_noise = edge_attributes.E_noise;
    auto& E_inliers = edge_attributes.E_inliers;
    auto& E_best_samples = edge_attributes.E_best_samples;

    // Use this data to decide if we want to estimate an essential matrix.
    const auto& F_inliers = edge_attributes.F_inliers;
    auto F_num_inliers = [&](const auto& ij) {
      return F_inliers[ij].vector().count();
    };
    // auto F_inlier_ratio = [&](const auto& ij) {
    //   return double(F_num_inliers(ij)) / F_inliers[ij].size();
    // };

    std::for_each(
        std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
          const auto& eij = edges[ij];
          const auto i = eij.first;
          const auto j = eij.second;
          const auto& Mij = matches[ij];
          const auto& ki = keypoints[i];
          const auto& kj = keypoints[j];
          const auto& Ki = view_attributes.cameras[i].K;
          const auto& Kj = view_attributes.cameras[j].K;
          const auto Ki_inv = Ki.inverse();
          const auto Kj_inv = Kj.inverse();

          SARA_DEBUG << "Calculating essential matrices between images:\n"
                     << "- image[" << i << "] = "  //
                     << view_attributes.group_names[i] << "\n"
                     << "- image[" << j << "] = "  //
                     << view_attributes.group_names[j] << "\n";

          SARA_DEBUG << "Internal camera matrices :\n"
                     << "- K[" << i << "] =\n"
                     << Ki << "\n"
                     << "- K[" << j << "] =\n"
                     << Kj << "\n";
          std::cout.flush();

          auto Eij = EssentialMatrix{};
          auto E_best_sample_ij = Tensor_<int, 1>{ESolver::num_points};
          auto E_inliers_ij = Tensor_<bool, 1>{static_cast<int>(Mij.size())};
          auto Fij = FundamentalMatrix{};
          if (F_num_inliers(ij) >= min_F_inliers)
          {
            // Estimate the fundamental matrix.
            std::tie(Eij, E_inliers_ij, E_best_sample_ij) =
                estimate_essential_matrix(Mij, ki, kj, Ki_inv, Kj_inv,
                                          num_samples, noise);

            Eij.matrix() = Eij.matrix().normalized();

            Fij.matrix() =
                Kj.inverse().transpose() * Eij.matrix() * Ki.inverse();

            SARA_DEBUG << "Eij = \n" << Eij << std::endl;
            SARA_DEBUG << "Fij = \n" << Fij << std::endl;
            SARA_CHECK(E_inliers_ij.row_vector());
            SARA_CHECK(E_best_sample_ij.row_vector());
          }
          else
          {
            Eij.matrix().setZero();
            E_best_sample_ij.flat_array().setZero();
            E_inliers_ij.flat_array().setZero();
          }

          if (debug)
          {
            const int display_step = 20;
            const auto& Ii = view_attributes.images[i];
            const auto& Ij = view_attributes.images[j];
            check_epipolar_constraints(Ii, Ij, Fij, Mij, E_best_sample_ij,
                                       E_inliers_ij, display_step,
                                       /* wait_key */ false);
          }

          // Update.
          E[ij] = Eij;
          E_inliers[ij] = E_inliers_ij;
          E_best_samples[ij] = E_best_sample_ij;
          // Useful if we use MLESAC and variants.
          E_noise[ij] = noise;
          // Useful if we use PROSAC sampling strategy.
          E_num_samples[ij] = num_samples;

          SARA_CHECK(E_num_samples[ij]);
          SARA_CHECK(E_noise[ij]);
        });

    h5_file.write_dataset("E", tensor_view(E), overwrite);
    h5_file.write_dataset("E_num_samples", tensor_view(E_num_samples),
                          overwrite);
    h5_file.write_dataset("E_noise", tensor_view(E_noise), overwrite);
    h5_file.write_dataset("E_best_samples", E_best_samples, overwrite);
    // Save E-edges.
    h5_file.get_group("E_inliers");
    std::for_each(std::begin(edge_ids), std::end(edge_ids),
                  [&](const auto& ij) {
                    const auto i = edges[ij].first;
                    const auto j = edges[ij].second;
                    h5_file.write_dataset(format("E_inliers/%d_%d", i, j),
                                          E_inliers[ij], overwrite);
                  });
  }

  auto inspect_essential_matrices(const std::string& dirpath,
                                  const std::string& h5_filepath,
                                  int display_step, bool wait_key) -> void
  {
    SARA_DEBUG << "Opening file " << h5_filepath << "..." << std::endl;
    auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};

    auto view_attributes = ViewAttributes{};

    // Load images (optional).
    SARA_DEBUG << "Listing images from:\n\t" << dirpath << std::endl;
    view_attributes.list_images(dirpath);
    view_attributes.read_images();

    // Load keypoints.
    SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath
               << std::endl;
    view_attributes.read_keypoints(h5_file);
    const auto& images = view_attributes.images;

    // Load the internal camera matrices from Strecha dataset.
    // N.B.: this is an ad-hoc code.
    SARA_DEBUG << "Reading internal camera matrices in Strecha's data format"
               << std::endl;
    std::for_each(std::begin(view_attributes.image_paths),
                  std::end(view_attributes.image_paths),
                  [&](const auto& image_path) {
                    const auto K_filepath =
                        dirpath + "/" + basename(image_path) + ".png.K";
                    SARA_DEBUG << "Reading internal camera matrix from:\n\t"
                               << K_filepath << std::endl;
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

    SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath
               << std::endl;
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

    std::for_each(
        std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
          const auto& eij = edges[ij];
          const auto i = eij.first;
          const auto j = eij.second;
          const auto& Mij = matches[ij];

          const auto& Eij = E[ij];

          const auto& Ki = view_attributes.cameras[i].K;
          const auto& Kj = view_attributes.cameras[j].K;

          SARA_DEBUG << "Internal camera matrices :\n"
                     << "- K[" << i << "] =\n"
                     << Ki << "\n"
                     << "- K[" << j << "] =\n"
                     << Kj << "\n";

          SARA_DEBUG
              << "Forming the fundamental matrix from the essential matrix:\n";
          std::cout.flush();
          auto Fij = FundamentalMatrix{};
          Fij.matrix() = Kj.inverse().transpose() * Eij.matrix() * Ki.inverse();

          SARA_DEBUG << "Fij = \n" << E[ij] << std::endl;
          SARA_CHECK(E_num_samples[ij]);
          SARA_CHECK(E_noise[ij]);
          SARA_CHECK(E_inliers[ij].row_vector());
          SARA_CHECK(E_inliers[ij].row_vector().count());
          SARA_CHECK(E_best_samples[ij].row_vector());

          const auto& Ii = images[i];
          const auto& Ij = images[j];
          check_epipolar_constraints(Ii, Ij, Fij, Mij, E_best_samples[ij],
                                     E_inliers[ij], display_step, wait_key);
        });
  }

} /* namespace DO::Sara */
