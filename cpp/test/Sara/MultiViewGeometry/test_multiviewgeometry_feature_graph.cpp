#define BOOST_TEST_MODULE "MultiViewGeometry/Geometry/Point Tracks"

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/StringFormat.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>

#include <boost/foreach.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/test/unit_test.hpp>


using namespace DO::Sara;


struct ConnectedComponents
{
  using Vertex = boost::graph_traits<FeatureGraph>::vertex_descriptor;
  using VertexIndex = boost::graph_traits<FeatureGraph>::vertices_size_type;

  using Rank = VertexIndex *;
  using Parent = Vertex *;

  using Components = boost::component_index<VertexIndex>;

  ConnectedComponents(FeatureGraph& graph)
    : g{graph}
    , rank{boost::num_vertices(graph)}
    , parent{boost::num_vertices(graph)}
  {
  }

  FeatureGraph& g;
  std::vector<Rank> rank;
  std::vector<Vertex> parent;
};

using DisjointSets = boost::disjoint_sets<ConnectedComponents::Rank,
                                          ConnectedComponents::Parent>;


BOOST_AUTO_TEST_CASE(test_feature_graph)
{
  auto g = FeatureGraph{};

  auto v1 = boost::add_vertex({0, 0}, g);
  auto v2 = boost::add_vertex({0, 1}, g);
  auto v3 = boost::add_vertex({1, 1}, g);
  auto v4 = boost::add_vertex({1, 1}, g);

  SARA_DEBUG << "g[v1] = " << format("(%d, %d)", g[v1].image_id, g[v1].local_id)
             << std::endl;
  SARA_DEBUG << "g[v2] = " << format("(%d, %d)", g[v2].image_id, g[v2].local_id)
             << std::endl;
  SARA_DEBUG << "g[v3] = " << format("(%d, %d)", g[v3].image_id, g[v3].local_id)
             << std::endl;
  SARA_DEBUG << "g[v4] = " << format("(%d, %d)", g[v4].image_id, g[v4].local_id)
             << std::endl;

  const auto [e1, b1] = boost::add_edge(v1, v4, g);
  const auto [e2, b2] = boost::add_edge(v2, v3, g);

  SARA_CHECK(e1);
  SARA_CHECK(e2);

  BOOST_CHECK(b1);
  BOOST_CHECK(b2);

  boost::print_graph(g, "");

  auto tracks = ConnectedComponents{g};
  //DisjointSets ds(tracks.rank[0], tracks.parent[0]);

  //boost::initialize_incremental_components(g, ds);
  //boost::incremental_components(g, ds);

  //ConnectedComponents::Components components(tracks.parent.begin(),
  //                                           tracks.parent.end());
  //// Iterate through the component indices
  //BOOST_FOREACH (ConnectedComponents::VertexIndex current_index, components)
  //{
  //  std::cout << "component " << current_index << " contains: ";

  //  // Iterate through the child vertex indices for [current_index]
  //  BOOST_FOREACH (ConnectedComponents::VertexIndex child_index,
  //                 components[current_index])
  //  {
  //    std::cout << child_index << " ";
  //  }

  //  std::cout << std::endl;
  //}
}
