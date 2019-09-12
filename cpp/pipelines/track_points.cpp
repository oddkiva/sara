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
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace DO::Sara;


auto track_points(const std::string& dirpath, const std::string& h5_filepath,
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

  // Load keypoints.
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  view_attributes.read_keypoints(h5_file);


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

  SARA_DEBUG << "Reading the two-view geometries..." << std::endl;
  edge_attributes.read_two_view_geometries(view_attributes, h5_file);

  // Convenient references.
  const auto& edge_ids = edge_attributes.edge_ids;
  const auto& edges = edge_attributes.edges;
  const auto& matches = edge_attributes.matches;

  const auto& E = edge_attributes.E;
  const auto& E_num_samples = edge_attributes.E_num_samples;
  const auto& E_noise = edge_attributes.E_noise;
  const auto& E_best_samples = edge_attributes.E_best_samples;
  const auto& E_inliers = edge_attributes.E_inliers;

  const auto& keypoints = view_attributes.keypoints;
  const auto num_keypoints =
      std::accumulate(std::begin(keypoints), std::end(keypoints), size_t(0),
                      [](const auto& sum, const auto& keys) -> size_t {
                        return sum + features(keys).size();
                      });
  SARA_CHECK(num_keypoints);

  const auto image_ids = range(num_vertices);
  SARA_CHECK(image_ids.row_vector());

  auto populate_gids = [&](auto image_id) {
    const auto num_features =
        static_cast<int>(features(keypoints[image_id]).size());
    auto lids = range(num_features);
    auto gids = std::vector<FeatureGID>(lids.size());
    std::transform(std::begin(lids), std::end(lids), std::begin(gids),
                   [&](auto lid) -> FeatureGID {
                     return {image_id, lid};
                   });
    return gids;
  };
  const auto gids = std::accumulate(
      std::begin(image_ids), std::end(image_ids), std::vector<FeatureGID>{},
      [&](const auto& gids, const auto image_id) {
        SARA_CHECK(image_id);
        auto gids_union = gids;
        ::append(gids_union, populate_gids(image_id));
        return gids_union;
      });
  SARA_CHECK(gids.size());

  // Populate the vertices.
  const auto feature_ids = range(static_cast<int>(gids.size()));
  auto graph = FeatureGraph{num_keypoints};
  std::for_each(std::begin(feature_ids), std::end(feature_ids),
                [&](auto v) { graph[v] = gids[v]; });

  auto feature_id_offset = std::vector<int>(num_vertices);
  feature_id_offset[0] = 0;
  for (auto i = 1; i < num_vertices; ++i)
    feature_id_offset[i] =
        feature_id_offset[i - 1] + features(keypoints[i - 1]).size();

  // Incremental connected components.
  using ICC = IncrementalConnectedComponentsHelper;
  auto rank = ICC::initialize_ranks(graph);
  auto parent = ICC::initialize_parents(graph);
  auto ds = ICC::initialize_disjoint_sets(rank, parent);
  ICC::initialize_incremental_components(graph, ds);

  auto add_edge = [&](auto u, auto v) {
    boost::add_edge(u, v, graph);
    ds.union_set(u, v);
  };

  auto find_sets = [&](const auto& graph, auto& ds) {
    for (auto [v, v_end] = boost::vertices(graph); v != v_end; ++v)
      std::cout << "representative[" << *v << "] = " << ds.find_set(*v)
                << std::endl;
    std::cout << std::endl;
  };

  auto print_graph = [&](const auto& graph) {
    std::cout << "An undirected graph:" << std::endl;
    boost::print_graph(graph, boost::get(boost::vertex_index, graph));
    std::cout << std::endl;
  };

  auto print_components = [&](const auto& components) {
    for (auto c : components)
    {
      std::cout << "component " << c << " contains: ";

      for (auto [child, child_end] = components[c]; child != child_end; ++child)
        std::cout << *child << " ";
      std::cout << std::endl;
    }
  };

  // Populate the edges.
  std::for_each(std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
    const auto& eij = edges[ij];
    const auto i = eij.first;
    const auto j = eij.second;
    const auto& Mij = matches[ij];
    const auto& inliers_ij = E_inliers[ij];
    const auto& cheirality_ij =
        edge_attributes.two_view_geometries[ij].cheirality;

    std::cout << std::endl;
    SARA_DEBUG << "Processing image pair " << i << " " << j << std::endl;

    SARA_DEBUG << "Checking if there are inliers..." << std::endl;
    SARA_CHECK(cheirality_ij.count());
    SARA_CHECK(inliers_ij.flat_array().count());
    if (inliers_ij.flat_array().count() == 0)
      return;

    SARA_DEBUG << "Calculating cheiral inliers..." << std::endl;
    SARA_CHECK(cheirality_ij.size());
    SARA_CHECK(inliers_ij.size());
    if (cheirality_ij.size() != inliers_ij.size())
        throw std::runtime_error{"cheirality_ij.size() != inliers_ij.size()"};

    const Array<bool, 1, Dynamic> cheiral_inliers =
        inliers_ij.row_vector().array() && cheirality_ij;
    SARA_CHECK(cheiral_inliers.size());
    SARA_CHECK(cheiral_inliers.count());

    // Convert each match 'm' to a pair of point indices '(p, q)'.
    SARA_DEBUG << "Transforming matches..." << std::endl;
    const auto pq_tensor = to_tensor(Mij);
    SARA_CHECK(Mij.size());
    SARA_CHECK(pq_tensor.size(0));

    if (pq_tensor.empty())
      return;

    SARA_DEBUG << "Updating disjoint sets..." << std::endl;
    for (int m = 0; m < pq_tensor.size(0); ++m)
    {
      if (!cheiral_inliers(m))
        continue;

      const auto p = pq_tensor(m, 0);
      const auto q = pq_tensor(m, 1);

      const auto &p_off = feature_id_offset[i];
      const auto &q_off = feature_id_offset[j];

      const auto vp = p_off + p;
      const auto vq = q_off + q;

#ifdef DEBUG
      SARA_CHECK(m);
      SARA_CHECK(p);
      SARA_CHECK(q);
      SARA_CHECK(p_off);
      SARA_CHECK(q_off);
      SARA_CHECK(vp);
      SARA_CHECK(vq);
#endif

      // Runtime checks.
      if (graph[vp].image_id != i)
        throw std::runtime_error{"image_id[vp] != i"};
      if (graph[vp].local_id != p)
        throw std::runtime_error{"local_id[vp] != p"};

      if (graph[vq].image_id != j)
        throw std::runtime_error{"image_id[vq] != j"};
      if (graph[vq].local_id != q)
        throw std::runtime_error{"local_id[vq] != q"};

      // Update the graph and the disjoint sets.
      add_edge(vp, vq);
    }
  });

  // Calculate the connected components.
  const auto components = ICC::get_components(parent);
  print_components(components);

  // Save the graph of features in HDF5 format.
  write_feature_graph(graph, h5_file, "feature_graph");


  // Prepare the bundle adjustment problem.
  auto components_filtered = std::vector<std::vector<int>>{};
  for (const auto& c : components)
  {
    int sz = 0;
    for (auto [v, v_end] = components[c]; v != v_end; ++v)
      ++sz;
    if (sz <= 1)
      continue;

    auto comp = std::vector<int>{};
    for (auto [v, v_end] = components[c]; v != v_end; ++v)
      comp.push_back(*v);

    components_filtered.push_back(comp);
  }

  const auto num_points = components_filtered.size();
  
  auto num_observations = 0;
  for (const auto& component : components_filtered)
    num_observations += component.size();

  const auto cameras = num_vertices;


  BundleAdjustmentProblem ba;
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
    const auto overwrite = vm.count("overwrite");
    const auto debug = vm.count("debug");

    track_points(dirpath, h5_filepath, overwrite, debug);

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
