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
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace sara = DO::Sara;

using namespace std;


namespace DO::Sara {

auto read_internal_camera_parameters(const std::string& filepath)
    -> Matrix3d
{
  std::ifstream file{filepath};
  if (!file)
    throw std::runtime_error{"File " + filepath + "does not exist!"};

  Matrix3d K;
  file >> K;

  return K;
}


auto match(const KeypointList<OERegion, float>& keys1,
           const KeypointList<OERegion, float>& keys2)
    -> std::vector<Match>
{
  AnnMatcher matcher{keys1, keys2, 0.6f};
  return matcher.compute_matches();
}

auto estimate_fundamental_matrix(const std::vector<Match>& Mij,
                                 const KeypointList<OERegion, float>& ki,
                                 const KeypointList<OERegion, float>& kj,
                                 int num_samples,
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

  auto estimator = EightPointAlgorithm{};
  auto distance = EpipolarDistance{};

  const auto [F, num_inliers, sample_best] =
      ransac(Mij_tensor, Pi, Pj, estimator, distance, num_samples, err_thres);

  SARA_CHECK(F);
  SARA_CHECK(num_inliers);
  SARA_CHECK(Mij.size());

  return std::make_tuple(F, num_inliers, sample_best);
}

auto estimate_essential_matrix(const std::vector<Match>& Mij,
                               const KeypointList<OERegion, float>& ki,
                               const KeypointList<OERegion, float>& kj,
                               const Matrix3d& Ki_inv,
                               const Matrix3d& Kj_inv,
                               int num_samples,
                               double err_thres)
{
  const auto to_double = [](const float& src) { return double(src); };
  const auto& fi = features(ki);
  const auto& fj = features(kj);
  const auto pi = extract_centers(fi).cwise_transform(to_double);
  const auto pj = extract_centers(fj).cwise_transform(to_double);

  const auto Pi = apply_transform(Ki_inv, homogeneous(pi));
  const auto Pj = apply_transform(Kj_inv, homogeneous(pj));

  const auto Mij_tensor = to_tensor(Mij);

  auto estimator = NisterFivePointAlgorithm{};
  auto distance = EpipolarDistance{};

  const auto [E, num_inliers, sample_best] =
      ransac(Mij_tensor, Pi, Pj, estimator, distance, num_samples, err_thres);

  SARA_CHECK(E);
  SARA_CHECK(num_inliers);
  SARA_CHECK(Mij.size());

  return std::make_tuple(E, num_inliers, sample_best);
}

auto check_epipolar_constraints(const Image<Rgb8>& Ii, const Image<Rgb8>& Ij,
                                const FundamentalMatrix& F,
                                const vector<Match>& Mij,
                                const Tensor_<int, 1>& sample_best,
                                double err_thres, int display_step)
{
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

  PairWiseDrawer drawer(Ii, Ij);
  drawer.set_viz_params(scale, scale, PairWiseDrawer::CatH);

  drawer.display_images();

  auto distance = EpipolarDistance{F.matrix()};

  for (size_t m = 0; m < Mij.size(); ++m)
  {
    const Vector3d X1 = Mij[m].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = Mij[m].y_pos().cast<double>().homogeneous();

    if (distance(X1, X2) > err_thres)
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

    const Vector3d X1 = Mij[sample_best(m)].x_pos().cast<double>().homogeneous();
    const Vector3d X2 = Mij[sample_best(m)].y_pos().cast<double>().homogeneous();

    const auto proj_X1 = F.right_epipolar_line(X1);
    const auto proj_X2 = F.left_epipolar_line(X2);

    // Draw the corresponding epipolar lines.
    drawer.draw_line_from_eqn(1, proj_X1.cast<float>(), Magenta8, 1);
    drawer.draw_line_from_eqn(0, proj_X2.cast<float>(), Magenta8, 1);
  }

  //get_key();
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


void match_keypoints(const std::string& dirpath, const std::string& h5_filepath)
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

  auto K_invs = std::vector<Eigen::Matrix3d>{};
  K_invs.reserve(group_names.size());
  std::transform(std::begin(group_names), std::end(group_names),
                 std::back_inserter(group_names),
                 [&](const std::string& group_name) {
                   return sara::read_internal_camera_parameters(
                              dirpath + "/" + group_name + ".png.K")
                       .inverse();
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

  auto matches = std::vector<std::vector<sara::Match>>{};
  matches.reserve(edges.size());
  std::transform(std::begin(edges), std::end(edges),
                 std::back_inserter(matches),
                 [&](const sara::EpipolarEdge& edge) {
                   const auto i = edge.i;
                   const auto j = edge.j;
                   return sara::match(keypoints[i], keypoints[j]);
                 });

  // Save matches to HDF5.
  auto edge_ids = sara::range(edges.size());
  std::for_each(
      std::begin(edge_ids), std::end(edge_ids), [&](const auto& edge_id) {
        const auto& edge = edges[edge_id];
        const auto i = edge.i;
        const auto j = edge.j;
        const auto& matches_ij = matches[edge_id];

        // Transform the data.
        auto Mij = std::vector<sara::IndexMatch>{};
        std::transform(
            std::begin(matches_ij), std::end(matches_ij),
            std::back_inserter(Mij), [](const auto& m) {
              return sara::IndexMatch{m.x_index(), m.y_index(), m.score()};
            });

        // Save the keypoints to HDF5
        const auto group_name = std::string{"matches"};
        h5_file.group(group_name);

        const auto match_dataset =
            group_name + "/" + std::to_string(i) + "_" + std::to_string(j);
        h5_file.write_dataset(match_dataset, tensor_view(Mij));
      });

  const auto num_samples = 1000;
  const auto f_err_thres = 5e-3;
  std::transform(
      std::begin(edge_ids), std::end(edge_ids), std::begin(edges),
      [&](const auto& edge_id) -> sara::EpipolarEdge {
        const auto& edge = edges[edge_id];
        const auto i = edge.i;
        const auto j = edge.j;
        const auto& Mij = matches[edge_id];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        // Estimate the fundamental matrix.
        const auto [F, num_inliers, sample_best] =
            sara::estimate_fundamental_matrix(Mij, ki, kj, num_samples,
                                              f_err_thres);

        return num_inliers < 100 ? sara::EpipolarEdge{i, j, Matrix3d::Zero()}
                                 : sara::EpipolarEdge{i, j, F.matrix()};
      });

  auto e_edges = edges;
  const auto e_err_thres = 5e-3;
  std::for_each(
      std::begin(edge_ids), std::end(edge_ids),
      [&](const auto& edge_id) {
        auto& edge = e_edges[edge_id];
        const auto i = edge.i;
        const auto j = edge.j;
        const auto& Mij = matches[edge_id];
        const auto& ki = keypoints[i];
        const auto& kj = keypoints[j];

        const auto& Ki_inv = K_invs[i];
        const auto& Kj_inv = K_invs[j];

        if (edge.m == Eigen::Matrix3d::Zero())
          return;

        const auto [E, num_inliers_e, sample_best_e] =
            sara::estimate_essential_matrix(Mij, ki, kj, Ki_inv, Kj_inv,
                                            num_samples, e_err_thres);

        edge.m = E;
      });

  for (int i = 0; i < N; ++i)
  {
    for (int j = i + 1; j < N; ++j)
    {
      // Specify the image pair to process.
      const auto& fi = image_paths[i];
      const auto& fj = image_paths[j];

      const auto gi = sara::basename(fi);
      const auto gj = sara::basename(fj);

      SARA_DEBUG << gi << std::endl;
      SARA_DEBUG << gj << std::endl;
      const auto Ii = sara::imread<sara::Rgb8>(fi);
      const auto Ij = sara::imread<sara::Rgb8>(fj);

      const auto Ki =
          sara::read_internal_camera_parameters(dirpath + "/" + gi + ".png.K");
      const auto Kj =
          sara::read_internal_camera_parameters(dirpath + "/" + gj + ".png.K");

      const Eigen::Matrix3d Ki_inv = Ki.inverse();
      const Eigen::Matrix3d Kj_inv = Kj.inverse();

      // Load the keypoints.
      const auto ki = sara::read_keypoints(h5_file, gi);
      const auto kj = sara::read_keypoints(h5_file, gj);


      // Match keypoints.
      const auto Mij = match(ki, kj);


      // Save the keypoints to HDF5
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


      const auto num_samples = 1000;
      const auto err_thres = 5e-3;

      //// Estimate the fundamental matrix.
      //const auto [F, num_inliers, sample_best] =
      //    sara::estimate_fundamental_matrix(Mij, ki, kj, num_samples,
      //                                      err_thres);

      const auto [E, num_inliers_e, sample_best_e] =
          sara::estimate_essential_matrix(Mij, ki, kj, Ki_inv, Kj_inv,
                                          num_samples, err_thres);

      auto F1 = sara::FundamentalMatrix{};
      F1.matrix() = Kj_inv.transpose() * E.matrix() * Ki_inv;

      // Visualize the estimated fundamental matrix.
      const int display_step = 20;
      //sara::check_epipolar_constraints(Ii, Ij, F, Mij, sample_best, err_thres,
      //                                 display_step);
      sara::check_epipolar_constraints(Ii, Ij, F1, Mij, sample_best_e,
                                       err_thres, display_step);
    }
  }
}


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Detect SIFT keypoints"};
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
    match_keypoints(dirpath, h5_filepath);

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
