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

#include <DO/Sara/Features/KeypointList.hpp>
#include <DO/Sara/MultiViewGeometry/DataTransformations.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>


namespace DO::Sara {

auto populate_feature_gids(
    const std::vector<KeypointList<OERegion, float>>& keypoints)
    -> std::vector<FeatureGID>
{
  const auto image_ids = range(static_cast<int>(keypoints.size()));

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

  const auto gids =
      std::accumulate(std::begin(image_ids), std::end(image_ids),  //
                      std::vector<FeatureGID>{},                   //
                      [&](const auto& gids, const auto image_id) {
                        auto gids_union = gids;
                        ::append(gids_union, populate_gids(image_id));
                        return gids_union;
                      });

  return gids;
}

auto calculate_feature_id_offsets(
    const std::vector<KeypointList<OERegion, float>>& keypoints)
    -> std::vector<int>
{
  auto fid_offsets = std::vector<int>(3, 0);
  std::transform(std::begin(keypoints), std::end(keypoints) - 1,
                 std::begin(fid_offsets) + 1, [](const auto& keypoints) {
                   return features(keypoints).size();
                 });

  std::partial_sum(std::begin(fid_offsets), std::end(fid_offsets),
                   std::begin(fid_offsets));

  return fid_offsets;
}

auto populate_feature_tracks(const ViewAttributes& view_attributes,
                             const EpipolarEdgeAttributes& epipolar_edges)
    -> std::pair<FeatureGraph, std::vector<std::vector<int>>>
{
  const auto& keypoints = view_attributes.keypoints;

  const auto gids = populate_feature_gids(keypoints);
  const auto num_keypoints = gids.size();

  // Populate the vertices.
  const auto feature_ids = range(static_cast<int>(num_keypoints));
  auto graph = FeatureGraph{num_keypoints};
  // Fill the GID attribute for each vertex.
  std::for_each(std::begin(feature_ids), std::end(feature_ids),
                [&](auto v) { graph[v] = gids[v]; });

  const auto feature_id_offset = calculate_feature_id_offsets(keypoints);

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

  const auto& edge_ids = epipolar_edges.edge_ids;
  const auto& edges = epipolar_edges.edges;
  const auto& matches = epipolar_edges.matches;
  const auto& E_inliers = epipolar_edges.E_inliers;
  const auto& two_view_geometries = epipolar_edges.two_view_geometries;

  // Populate the edges.
  std::for_each(std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
    const auto& eij = edges[ij];
    const auto i = eij.first;
    const auto j = eij.second;
    const auto& Mij = matches[ij];
    const auto& inliers_ij = E_inliers[ij];
    const auto& cheirality_ij = two_view_geometries[ij].cheirality;

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
  auto components = std::vector<std::vector<int>>{};
  {
    const auto components_tmp = ICC::get_components(parent);
    components.resize(components_tmp.size());
    for (auto c : components_tmp)
      for (auto [child, child_end] = components_tmp[c]; child != child_end; ++child)
        components[c].push_back(static_cast<int>(*child));
  }


  return {graph, components};
}

auto filter_feature_tracks(const FeatureGraph& graph,
                           const std::vector<std::vector<int>>& components)
    -> std::set<std::set<FeatureGID>>
{
  auto feature_tracks_filtered = std::set<std::set<FeatureGID>>{};
  for (const auto& component: components)
  {
    auto feature_track = std::set<FeatureGID>{};
    std::transform(component.begin(), component.end(),
                   std::inserter(feature_track, std::begin(feature_track)),
                   [&](const auto v) { return graph[v]; });

    // We are only interested in features that have correspondences.
    // So we must collect only components of size 2 at least.
    if (feature_track.size() >= 2)
      feature_tracks_filtered.insert(feature_track);
  }

  return feature_tracks_filtered;
}

template <>
struct CalculateH5Type<FeatureGID>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(FeatureGID)};
    INSERT_MEMBER(h5_comp_type, FeatureGID, image_id);
    INSERT_MEMBER(h5_comp_type, FeatureGID, local_id);
    return h5_comp_type;
  }
};


auto write_feature_graph(const FeatureGraph& graph, H5File& file,
                                const std::string& group_name) -> void
{
  auto features = std::vector<FeatureGID>(boost::num_vertices(graph));
  for (auto [v, v_end] = boost::vertices(graph); v != v_end; ++v)
    features[*v] = {graph[*v].image_id, graph[*v].local_id};

  auto matches = std::vector<Vector2i>{};
  for (auto [e, e_end] = boost::edges(graph); e != e_end; ++e)
    matches.push_back({boost::source(*e, graph), boost::target(*e, graph)});

  file.get_group(group_name);
  file.write_dataset(group_name + "/" + "features", tensor_view(features));
  file.write_dataset(group_name + "/" + "matches", tensor_view(matches));
}


auto read_feature_graph(H5File& file, const std::string& group_name)
    -> FeatureGraph
{
  auto features = std::vector<FeatureGID>{};
  auto matches = std::vector<Vector2i>{};

  file.read_dataset(group_name + "/" + "features", features);
  file.read_dataset(group_name + "/" + "matches", matches);

  // Reconstruct the graph.
  auto g = FeatureGraph{};

  for (const auto& v : features)
    boost::add_vertex(v, g);

  for (const auto& e : matches)
    boost::add_edge(e(0), e(1), g);

  return g;
}

} /* namespace DO::Sara */
