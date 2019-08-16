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
#include <DO/Sara/Features/Draw.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry.hpp>
#include <DO/Sara/SfM/BuildingBlocks/FundamentalMatrixEstimation.hpp>

#include <boost/filesystem.hpp>


namespace fs = boost::filesystem;


namespace DO::Sara {

using FSolver = EightPointAlgorithm;

auto estimate_fundamental_matrix(
    const std::vector<Match>& Mij,
    const KeypointList<OERegion, float>& ki,
    const KeypointList<OERegion, float>& kj, int num_samples,
    double err_thres)
  -> std::tuple<FundamentalMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>
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
  auto distance = EpipolarDistance{};

  const auto [F, inliers, sample_best] = ransac(
      Mij_tensor, Pi, Pj, estimator, distance, num_samples, err_thres);

#ifdef DEBUG
  SARA_CHECK(F);
  SARA_CHECK(inliers.row_vector());
  SARA_CHECK(inliers.row_vector());
  SARA_CHECK(Mij.size());
#endif

  return std::make_tuple(F, inliers, sample_best);
}

auto estimate_fundamental_matrices(const std::string& dirpath,
                                   const std::string& h5_filepath,
                                   bool overwrite,
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

  // Load keypoints (optional).
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  view_attributes.read_keypoints(h5_file);
  const auto& keypoints = view_attributes.keypoints;

  // Initialize the epipolar graph.
  const auto num_vertices = int(view_attributes.image_paths.size());
  SARA_CHECK(num_vertices);

  auto edge_attributes = EpipolarEdgeAttributes{};
  SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
  edge_attributes.initialize_edges(num_vertices);

  SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath << std::endl;
  edge_attributes.read_matches(h5_file, view_attributes);

  SARA_DEBUG << "Preallocate the F data structures..." << std::endl;
  edge_attributes.resize_fundamental_edge_list();

  const auto& edge_ids = edge_attributes.edge_ids;
  const auto& edges = edge_attributes.edges;
  const auto& matches = edge_attributes.matches;
  SARA_CHECK(edge_ids.size());
  SARA_CHECK(edges.size());
  SARA_CHECK(matches.size());


  // Mutate these.
  auto& F = edge_attributes.F;
  auto& F_num_samples = edge_attributes.F_num_samples;
  auto& F_noise = edge_attributes.F_noise;
  auto& F_inliers = edge_attributes.F_inliers;
  auto& F_best_samples = edge_attributes.F_best_samples;

  const auto num_samples = 1000;
  const auto f_err_thres = 5e-3;
  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;
        const auto& Mij = matches[ij];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        SARA_DEBUG << "Calculating fundamental matrices between images:\n"
                   << "- image[" << i << "] = "  //
                   << view_attributes.group_names[i] << "\n"
                   << "- image[" << j << "] = "  //
                   << view_attributes.group_names[j] << "\n";
        std::cout.flush();

        // Estimate the fundamental matrix.
        const auto [Fij, F_inliers_ij, F_best_sample_ij] =
            estimate_fundamental_matrix(Mij, ki, kj, num_samples, f_err_thres);
        SARA_DEBUG << "Fij = \n" << Fij << std::endl;
        SARA_CHECK(F_inliers_ij.row_vector());
        SARA_CHECK(F_best_sample_ij.row_vector());

        if (debug)
        {
          const int display_step = 20;
          const auto& Ii = view_attributes.images[i];
          const auto& Ij = view_attributes.images[j];
          check_epipolar_constraints(Ii, Ij, Fij, Mij, F_best_sample_ij,
                                     F_inliers_ij, display_step);
        }

        // Update.
        F[ij] = Fij;
        F_inliers[ij] = F_inliers_ij;
        F_best_samples[ij] = F_best_sample_ij;
        F_noise[ij] = f_err_thres;
      });

  // Save fundamental matrices and additional info from RANSAC.
  h5_file.write_dataset("F", tensor_view(F), overwrite);
  h5_file.write_dataset("F_num_samples", tensor_view(F_num_samples),
                        overwrite);
  h5_file.write_dataset("F_noise", tensor_view(F_noise), overwrite);
  h5_file.write_dataset("F_best_samples", F_best_samples, overwrite);

  h5_file.get_group("F_inliers");
  std::for_each(std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
                const auto i = edges[ij].first;
                const auto j = edges[ij].second;
                h5_file.write_dataset(format("F_inliers/%d_%d", i, j),
                                      F_inliers[ij], overwrite);
  });
}

auto check_epipolar_constraints(const Image<Rgb8>& Ii, const Image<Rgb8>& Ij,
                                const FundamentalMatrix& F,
                                const std::vector<Match>& Mij,
                                const TensorView_<int, 1>& sample_best,
                                const TensorView_<bool, 1>& inliers,
                                int display_step, bool wait_key) -> void
{
  const auto scale = 0.25f;
  const auto w = int((Ii.width() + Ij.width()) * scale + 0.5f);
  const auto h = int(std::max(Ii.height(), Ij.height()) * scale + 0.5f);

  if (!active_window())
  {
    create_window(w, h);
    set_antialiasing();
  }

  if (get_sizes(active_window()) != Eigen::Vector2i(w, h))
    resize_window(w, h);

  PairWiseDrawer drawer(Ii, Ij);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

  drawer.display_images();

  for (size_t m = 0; m < Mij.size(); ++m)
  {
    const Eigen::Vector3d X1 = Mij[m].x_pos().cast<double>().homogeneous();
    const Eigen::Vector3d X2 = Mij[m].y_pos().cast<double>().homogeneous();

    if (!inliers(m))
      continue;

    if (m % display_step == 0)
    {
      drawer.draw_match(Mij[m], Blue8, false);

      const auto proj_X1 = F.right_epipolar_line(X1);
      const auto proj_X2 = F.left_epipolar_line(X2);

      drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Cyan8, 1);
      drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Cyan8, 1);
    }
  }

  for (size_t m = 0; m < sample_best.size(); ++m)
  {
    // Draw the best elemental subset drawn by RANSAC.
    drawer.draw_match(Mij[sample_best(m)], Red8, true);

    const Eigen::Vector3d X1 =
        Mij[sample_best(m)].x_pos().cast<double>().homogeneous();
    const Eigen::Vector3d X2 =
        Mij[sample_best(m)].y_pos().cast<double>().homogeneous();

    const auto proj_X1 = F.right_epipolar_line(X1);
    const auto proj_X2 = F.left_epipolar_line(X2);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Magenta8, 1);
  }

  if (wait_key)
    get_key();
}

auto inspect_fundamental_matrices(const std::string& dirpath,
                                  const std::string& h5_filepath,
                                  int display_step,
                                  bool wait_key) -> void
{
  SARA_DEBUG << "Opening file " << h5_filepath << "..." << std::endl;
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};

  auto view_attributes = ViewAttributes{};

  // Load images (optional).
  SARA_DEBUG << "Listing images from:\n\t" << dirpath << std::endl;
  view_attributes.list_images(dirpath);
  view_attributes.read_images();

  // Load keypoints (optional).
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  view_attributes.read_keypoints(h5_file);
  const auto& images = view_attributes.images;

  // Initialize the epipolar graph.
  const auto num_vertices = int(view_attributes.image_paths.size());
  SARA_CHECK(num_vertices);

  auto edge_attributes = EpipolarEdgeAttributes{};
  SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
  edge_attributes.initialize_edges(num_vertices);

  SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath << std::endl;
  edge_attributes.read_matches(h5_file, view_attributes);

  SARA_DEBUG << "Reading the fundamental matrices..." << std::endl;
  edge_attributes.resize_fundamental_edge_list();
  edge_attributes.read_fundamental_matrices(view_attributes, h5_file);

  // Convenient references.
  const auto& edge_ids = edge_attributes.edge_ids;
  const auto& edges = edge_attributes.edges;
  const auto& matches = edge_attributes.matches;

  const auto& F = edge_attributes.F;
  const auto& F_num_samples = edge_attributes.F_num_samples;
  const auto& F_noise = edge_attributes.F_noise;
  const auto& F_best_samples = edge_attributes.F_best_samples;
  const auto& F_inliers = edge_attributes.F_inliers;

  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;
        const auto& Mij = matches[ij];

        SARA_DEBUG << "Fij = \n" << F[ij] << std::endl;
        SARA_CHECK(F_num_samples[ij]);
        SARA_CHECK(F_noise[ij]);
        SARA_CHECK(F_inliers[ij].row_vector());
        SARA_CHECK(F_inliers[ij].row_vector().count());
        SARA_CHECK(F_best_samples[ij].row_vector());

        const auto& Ii = images[i];
        const auto& Ij = images[j];
        check_epipolar_constraints(Ii, Ij, F[ij], Mij, F_best_samples[ij],
                                   F_inliers[ij], display_step, wait_key);
      });
}

} /* namespace DO::Sara */
