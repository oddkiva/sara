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
  std::vector<VertexIndex> rank;
  std::vector<Vertex> parent;
};

using DisjointSets = boost::disjoint_sets<ConnectedComponents::Rank,
                                          ConnectedComponents::Parent>;

BOOST_AUTO_TEST_CASE(test_connected_components)
{
  const int N = 6;
  FeatureGraph G(N);
  boost::add_edge(0, 1, G);
  boost::add_edge(1, 4, G);
  boost::add_edge(4, 0, G);
  boost::add_edge(2, 5, G);

  std::vector<int> c(num_vertices(G));
  int num = boost::connected_components(
      G, make_iterator_property_map(c.begin(),
                                    boost::get(boost::vertex_index, G), c[0]));

  std::cout << std::endl;
  std::cout << "Total number of components: " << num << std::endl;
  for (auto i = c.begin(); i != c.end(); ++i)
    std::cout << "Vertex " << i - c.begin() << " is in component " << *i
              << std::endl;
  std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(test_incremental_connected_components)
{
  using Vertex = boost::graph_traits<FeatureGraph>::vertex_descriptor;
  using VertexIndex = boost::graph_traits<FeatureGraph>::vertices_size_type;

  using Rank = VertexIndex*;
  using Parent = Vertex*;

  // constexpr auto VERTEX_COUNT = 6;
  // auto graph = FeatureGraph(VERTEX_COUNT);
  auto graph = FeatureGraph{};
  auto v0 = boost::add_vertex({0, 0}, graph);
  auto v1 = boost::add_vertex({0, 1}, graph);
  auto v2 = boost::add_vertex({0, 2}, graph);
  auto v3 = boost::add_vertex({1, 0}, graph);
  auto v4 = boost::add_vertex({1, 1}, graph);
  auto v5 = boost::add_vertex({1, 2}, graph);

  auto rank = std::vector<VertexIndex>(num_vertices(graph));
  auto parent = std::vector<Vertex>(num_vertices(graph));

  boost::disjoint_sets<Rank, Parent> ds(&rank[0], &parent[0]);

  boost::initialize_incremental_components(graph, ds);
  boost::incremental_components(graph, ds);

  boost::graph_traits<FeatureGraph>::edge_descriptor edge;
  bool flag;

  boost::tie(edge, flag) = boost::add_edge(v0, v1, graph);
  ds.union_set(v0, v1);

  boost::tie(edge, flag) = boost::add_edge(v1, v4, graph);
  ds.union_set(v1, v4);

  boost::tie(edge, flag) = boost::add_edge(v4, v0, graph);
  ds.union_set(v4, v0);

  boost::tie(edge, flag) = boost::add_edge(v2, v5, graph);
  ds.union_set(v2, v5);

  std::cout << "An undirected graph:" << std::endl;
  boost::print_graph(graph, boost::get(boost::vertex_index, graph));
  std::cout << std::endl;

  for (auto [v, v_end] = boost::vertices(graph); v != v_end; ++v) {
    std::cout << "representative[" << *v << "] = " <<
    ds.find_set(*v) << std::endl;
  }

  std::cout << std::endl;

  using Components = boost::component_index<VertexIndex>;

  Components components(parent.begin(), parent.end());

  // Iterate through the component indices
  for (auto c: components) {
    std::cout << "component " << c << " contains: ";

    // Iterate through the child vertex indices for [c]
    for(auto [child, child_end] = components[c]; child != child_end; ++child) {
      std::cout << *child << " ";
    }

    std::cout << std::endl;
  }
}
