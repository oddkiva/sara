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
#include <DO/Sara/Core/StdVectorHelpers.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Features/Draw.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Match.hpp>
#include <DO/Sara/Match/IndexMatch.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>

#include <iostream>


namespace fs = boost::filesystem;


namespace DO::Sara {

auto detect_keypoints(const std::string& dirpath,
                      const std::string& h5_filepath,  //
                      bool overwrite) -> void
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_TRUNC};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = imread<float>(path);

        SARA_DEBUG << "Computing SIFT keypoints " << path << "..." << std::endl;
        const auto keys = compute_sift_keypoints(image);

        const auto group_name = basename(path);
        h5_file.get_group(group_name);

        SARA_DEBUG << "Saving SIFT keypoints of " << path << "..." << std::endl;
        write_keypoints(h5_file, group_name, keys, overwrite);
      });
}


auto read_keypoints(const std::string& dirpath, const std::string& h5_filepath)
    -> void
{
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDONLY};
  auto image_paths = std::vector<std::string>{};
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));

  std::for_each(
      std::begin(image_paths), std::end(image_paths), [&](const auto& path) {
        SARA_DEBUG << "Reading image " << path << "..." << std::endl;
        const auto image = imread<float>(path);

        const auto group_name = basename(path);

        SARA_DEBUG << "Read keypoints for " << group_name << "..." << std::endl;
        const auto keys = read_keypoints(h5_file, group_name);

        const auto& features = std::get<0>(keys);

        // Visual inspection.
        if (!active_window())
        {
          create_window(image.sizes() / 2, group_name);
          set_antialiasing();
        }

        if (get_sizes(active_window()) != image.sizes() / 2)
          resize_window(image.sizes() / 2);

        display(image, 0, 0, 0.5);
        draw_oe_regions(features, Red8, 0.5f);
        get_key();
      });

  if (active_window())
    close_window();
}


auto match(const KeypointList<OERegion, float>& keys1,
           const KeypointList<OERegion, float>& keys2,
           float lowe_ratio)
    -> std::vector<Match>
{
  AnnMatcher matcher{keys1, keys2, lowe_ratio};
  return matcher.compute_matches();
}


auto match_keypoints(const std::string& dirpath, const std::string& h5_filepath,
                     bool overwrite) -> void
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    cp(h5_filepath, h5_filepath + ".bak");

  auto h5_file = H5File{h5_filepath, H5F_ACC_RDWR};

  auto image_paths = std::vector<std::string>{};
  append(image_paths, ls(dirpath, ".png"));
  append(image_paths, ls(dirpath, ".jpg"));
  std::sort(image_paths.begin(), image_paths.end());

  auto group_names = std::vector<std::string>{};
  group_names.reserve(image_paths.size());
  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::back_inserter(group_names),
                 [&](const std::string& image_path) {
                   return basename(image_path);
                 });

  auto keypoints = std::vector<KeypointList<OERegion, float>>{};
  keypoints.reserve(image_paths.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(keypoints),
                 [&](const std::string& group_name) {
                   return read_keypoints(h5_file, group_name);
                 });

  const auto N = int(image_paths.size());
  auto edges = std::vector<std::pair<int, int>>{};
  edges.reserve(N * (N - 1) / 2);
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j)
      edges.emplace_back(i, j);

  auto matches = std::vector<std::vector<Match>>{};
  matches.reserve(edges.size());
  std::transform(std::begin(edges), std::end(edges),
                 std::back_inserter(matches),
                 [&](const auto& edge) {
                   const auto i = edge.first;
                   const auto j = edge.second;
                   return match(keypoints[i], keypoints[j]);
                 });

  // Save matches to HDF5.
  auto edge_ids = range(edges.size());
  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& e) {
        const auto& ij = edges[e];
        const auto i = ij.first;
        const auto j = ij.second;
        const auto& matches_ij = matches[e];

        // Transform the data.
        auto Mij = std::vector<IndexMatch>{};
        std::transform(
            std::begin(matches_ij), std::end(matches_ij),
            std::back_inserter(Mij), [](const auto& m) {
              return IndexMatch{m.x_index(), m.y_index(), m.score()};
            });

        // Save the keypoints to HDF5
        const auto group_name = std::string{"matches"};
        h5_file.get_group(group_name);

        const auto match_dataset =
            group_name + "/" + std::to_string(i) + "_" + std::to_string(j);
        h5_file.write_dataset(match_dataset, tensor_view(Mij), overwrite);
      });
}



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


using ESolver = NisterFivePointAlgorithm;

auto estimate_essential_matrix(
    const std::vector<Match>& Mij,
    const KeypointList<OERegion, float>& ki,
    const KeypointList<OERegion, float>& kj,
    const Eigen::Matrix3d& Ki_inv, const Eigen::Matrix3d& Kj_inv,
    int num_samples, double err_thres)
  -> std::tuple<EssentialMatrix, Tensor_<bool, 1>, Tensor_<int, 1>>
{
  const auto to_double = [](const float& src) { return double(src); };
  const auto& fi = features(ki);
  const auto& fj = features(kj);
  const auto ui = extract_centers(fi).cwise_transform(to_double);
  const auto uj = extract_centers(fj).cwise_transform(to_double);

  const auto uni = apply_transform(Ki_inv, homogeneous(ui));
  const auto unj = apply_transform(Kj_inv, homogeneous(uj));

  const auto Mij_tensor = to_tensor(Mij);

  auto estimator = ESolver{};
  auto distance = EpipolarDistance{};

  const auto [E, inliers, sample_best] =
      ransac(Mij_tensor, uni, unj, estimator, distance, num_samples, err_thres);

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
  auto F_inlier_ratio = [&](const auto& ij) {
    return double(F_num_inliers(ij)) / F_inliers[ij].size();
  };

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
                   << "- K[" << i << "] =\n" << Ki << "\n"
                   << "- K[" << j << "] =\n" << Kj << "\n";
        std::cout.flush();

        auto Eij = EssentialMatrix{};
        auto E_best_sample_ij = Tensor_<int, 1>{ESolver::num_points};
        auto E_inliers_ij = Tensor_<bool, 1>{Mij.size()};
        auto Fij = FundamentalMatrix{};
        if (F_num_inliers(ij) >= min_F_inliers)
        {
          // Estimate the fundamental matrix.
          std::tie(Eij, E_inliers_ij, E_best_sample_ij) =
              estimate_essential_matrix(Mij, ki, kj, Ki_inv, Kj_inv,
                                        num_samples, noise);

          Eij.matrix() = Eij.matrix().normalized();

          Fij.matrix() = Kj.inverse().transpose() * Eij.matrix() * Ki.inverse();

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
  std::for_each(std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
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

  // Load keypoints (optional).
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  view_attributes.read_keypoints(h5_file);
  const auto& images = view_attributes.images;

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

  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
        const auto& eij = edges[ij];
        const auto i = eij.first;
        const auto j = eij.second;
        const auto& Mij = matches[ij];

        const auto& Eij = E[ij];

        const auto& Ki = view_attributes.cameras[i].K;
        const auto& Kj = view_attributes.cameras[j].K;
        const auto Ki_inv = Ki.inverse();
        const auto Kj_inv = Kj.inverse();

        SARA_DEBUG << "Internal camera matrices :\n"
                   << "- K[" << i << "] =\n" << Ki << "\n"
                   << "- K[" << j << "] =\n" << Kj << "\n";

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
