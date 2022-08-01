#include "SquareReconstruction.hpp"


static auto find_next_square_vertex(  //
    int edge, int current_vertex,     //
    const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge) -> int
{
  struct Vertex
  {
    int id;
    float score;
    auto operator<(const Vertex& other) const
    {
      return score < other.score;
    }
  };

  const auto& cs = corners_adjacent_to_edge[edge];
  auto vs = std::vector<Vertex>(corners_adjacent_to_edge[edge].size());
  std::transform(cs.begin(), cs.end(), vs.begin(),
                 [&corners](const auto& v) -> Vertex {
                   return {v, corners[v].score};
                 });
  std::sort(vs.rbegin(), vs.rend());

  for (const auto& v : vs)
  {
    if (v.id == current_vertex)
      continue;

    const auto& a = corners[current_vertex].coords;
    const auto& b = corners[v.id].coords;

    auto rotation = Eigen::Matrix2f{};
    rotation.col(0) = edge_grads[edge].normalized();
    rotation.col(1) = (b - a).normalized();
    if (rotation.determinant() > 0.8f)
      return v.id;
  }
  return -1;
}


auto reconstruct_black_square_from_corner(
    int c,                //
    int start_direction,  //
    const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>
{
  auto square = std::array<int, 4>{};
  std::fill(square.begin(), square.end(), -1);

  // Initialize the square with the seed corner.
  square[0] = c;

  // There is always be 4 edges from the seed corner, otherwise it's a bug.
  if (edges_adjacent_to_corner[c].size() != 4)
    throw std::runtime_error{"There must be exactly 4 adjacent edges from the seed corner!"};
  auto edge_it = edges_adjacent_to_corner[square[0]].begin();
  for (auto it = 0; it < start_direction; ++it)
    ++edge_it;
  auto edge = *edge_it;

  for (auto i = 1; i < 4; ++i)
  {
    square[i] = find_next_square_vertex(edge, square[i - 1],  //
                                            corners, edge_grads,  //
                                            corners_adjacent_to_edge);
    if (square[i] == -1)
      return std::nullopt;

    // Choose the next edge: the next edge is the one that forms the rightmost
    // angle with the previous edge.
    const auto& edges = edges_adjacent_to_corner[square[i]];
    auto next_edge = -1;
    auto det = -1.f;
    for (const auto& e: edges)
    {
      if (e == edge)
        continue;

      auto rotation = Eigen::Matrix2f{};
      rotation.col(0) = edge_grads[edge].normalized();
      rotation.col(1) = edge_grads[e].normalized();
      const auto d = rotation.determinant();
      if (d > det && d > 0) // Important check the sign.
      {
        next_edge = e;
        det = d;
      }
    }
    if (det <= 0)
      return std::nullopt;
    edge = next_edge;
  }

  // I want unambiguously good squares.
  if (square[0] != c)
    return std::nullopt;  // Ambiguity.

  // Reorder the square vertices as follows:
  const auto vmin = std::min_element(square.begin(), square.end());
  if (vmin != square.begin())
    std::rotate(square.begin(), vmin, square.end());

  return square;
}

auto reconstruct_black_square_from_corner(
    const int c, const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>
{
  auto square = std::optional<std::array<int, 4>>{};
  for (auto d = 0; d < 4; ++d)
  {
    square = reconstruct_black_square_from_corner(c, d, corners, edge_grads,
                                                  edges_adjacent_to_corner,
                                                  corners_adjacent_to_edge);
    if (square != std::nullopt)
      return square;
  }
  return std::nullopt;
}


auto reconstruct_white_square_from_corner(
    int c,                //
    int start_direction,  //
    const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>
{
  auto square = std::array<int, 4>{};
  std::fill(square.begin(), square.end(), -1);

  // Initialize the square with the seed corner.
  square[0] = c;

  // There is always be 4 edges from the seed corner, otherwise it's a bug.
  if (edges_adjacent_to_corner[c].size() != 4)
    throw std::runtime_error{"There must be exactly 4 adjacent edges from the seed corner!"};
  auto edge_it = edges_adjacent_to_corner[square[0]].begin();
  for (auto it = 0; it < start_direction; ++it)
    ++edge_it;
  auto edge = *edge_it;

  for (auto i = 1; i < 4; ++i)
  {
    square[i] = find_next_square_vertex(edge, square[i - 1],  //
                                        corners, edge_grads,  //
                                        corners_adjacent_to_edge);
    if (square[i] == -1)
      return std::nullopt;

    // The next edge is the one that form the rightmost angle.
    const auto& edges = edges_adjacent_to_corner[square[i]];
    auto next_edge = -1;
    auto det = 1.f;
    for (const auto& e: edges)
    {
      if (e == edge)
        continue;
      auto rotation = Eigen::Matrix2f{};
      rotation.col(0) = edge_grads[edge].normalized();
      rotation.col(1) = edge_grads[e].normalized();
      const auto d = rotation.determinant();
      if (d < det && d < 0) // Important check the sign.
      {
        next_edge = e;
        det = d;
      }
    }
    if (det >= 0)
      return std::nullopt;
    edge = next_edge;
  }

  // I want unambiguously good squares.
  if (square[0] != c)
    return std::nullopt;  // Ambiguity.

  // Reorder the square vertices as follows:
  const auto vmin = std::min_element(square.begin(), square.end());
  if (vmin != square.begin())
    std::rotate(square.begin(), vmin, square.end());

  return square;
}

auto reconstruct_white_square_from_corner(
    const int c, const std::vector<Corner<float>>& corners,
    const std::vector<Eigen::Vector2f>& edge_grads,
    const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
    const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
    -> std::optional<std::array<int, 4>>
{
  auto square = std::optional<std::array<int, 4>>{};
  for (auto d = 0; d < 4; ++d)
  {
    square = reconstruct_white_square_from_corner(c, d, corners, edge_grads,
                                                  edges_adjacent_to_corner,
                                                  corners_adjacent_to_edge);
    if (square != std::nullopt)
      return square;
  }
  return std::nullopt;
}
