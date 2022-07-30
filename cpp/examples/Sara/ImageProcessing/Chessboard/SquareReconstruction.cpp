#include "SquareReconstruction.hpp"


static const auto dirs = std::array{
    Direction::Up,
    Direction::Right,
    Direction::Down,
    Direction::Left,
};


auto direction_type(const float angle) -> Direction
{
  static constexpr auto pi = static_cast<float>(M_PI);
  static constexpr auto pi_over_4 = static_cast<float>(M_PI / 4.);
  static constexpr auto three_pi_over_4 = static_cast<float>(3. * M_PI / 4.);
  static constexpr auto five_pi_over_4 = static_cast<float>(5. * M_PI / 4.);
  static constexpr auto seven_pi_over_4 = static_cast<float>(7. * M_PI / 4.);
  if (-pi <= angle && angle < -three_pi_over_4)
    return Direction::Left;
  if (-three_pi_over_4 <= angle && angle < -pi_over_4)
    return Direction::Up;
  if (-pi_over_4 <= angle && angle < pi_over_4)
    return Direction::Right;
  if (pi_over_4 <= angle && angle < three_pi_over_4)
    return Direction::Down;
  if (three_pi_over_4 <= angle && angle < five_pi_over_4)
    return Direction::Left;
  if (five_pi_over_4 <= angle && angle < seven_pi_over_4)
    return Direction::Up;
  else  // seven_pi_over_4 -> two_pi
    return Direction::Right;
}

static auto find_edge(const std::unordered_set<int>& edges, Direction what,
                      const std::vector<Eigen::Vector2f>& edge_grads)
{
  for (const auto& e : edges)
    if (direction_type(edge_grads[e]) == what)
      return e;
  return -1;
};

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

  square[0] = c;

  for (auto i = 1; i < 4; ++i)
  {
    const auto dir = dirs[(i - 1 + start_direction) % 4];
    const auto edge =
        find_edge(edges_adjacent_to_corner[square[i - 1]], dir, edge_grads);
    if (edge == -1)
      return std::nullopt;

    square[i % 4] = find_next_square_vertex(edge, square[i - 1],  //
                                            corners, edge_grads,  //
                                            corners_adjacent_to_edge);
    if (square[i % 4] == -1)
      return std::nullopt;
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
  static const auto dirs = std::array{
      Direction::Up,
      Direction::Right,
      Direction::Down,
      Direction::Left,
  };

  auto square = std::array<int, 4>{};
  std::fill(square.begin(), square.end(), -1);

  square[0] = c;

  for (auto i = 1; i < 4; ++i)
  {
    const auto dir = dirs[((4 - i + 1) + start_direction) % 4];
    const auto edge =
        find_edge(edges_adjacent_to_corner[square[i - 1]], dir, edge_grads);
    if (edge == -1)
      return std::nullopt;

    square[i % 4] = find_next_square_vertex(edge, square[i - 1],  //
                                            corners, edge_grads,  //
                                            corners_adjacent_to_edge);
    if (square[i % 4] == -1)
      return std::nullopt;
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
