#include "SquareReconstruction.hpp"


namespace DO::Sara {

  static auto find_next_square_vertex(  //
      int edge, int current_vertex,     //
      const std::vector<Corner<float>>& corners,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> int
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
      rotation.col(0) = edge_grad_mean[edge].normalized();
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
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
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
      throw std::runtime_error{
          "There must be exactly 4 adjacent edges from the seed corner!"};
    auto edge_it = edges_adjacent_to_corner[square[0]].begin();
    for (auto it = 0; it < start_direction; ++it)
      ++edge_it;
    auto edge = *edge_it;

    // The edge must be straight otherwise it's over.
    {
      const auto& grad_cov = edge_grad_cov[edge];
      static constexpr auto kappa = 0.2f;
      using DO::Sara::square;
      const auto cornerness = grad_cov.determinant() -  //
                              kappa * square(grad_cov.trace());
      if (cornerness > 0)
        return std::nullopt;
    }

    // Pursue the square reconstruction.
    for (auto i = 1; i < 4; ++i)
    {
      square[i] = find_next_square_vertex(edge, square[i - 1],  //
                                          corners, edge_grad_mean,
                                          corners_adjacent_to_edge);
      if (square[i] == -1)
        return std::nullopt;

      // Choose the next edge: the next edge is the one that forms the rightmost
      // angle with the previous edge.
      const auto& edges = edges_adjacent_to_corner[square[i]];
      auto next_edge = -1;
      auto det = -1.f;
      for (const auto& e : edges)
      {
        if (e == edge)
          continue;

        // The edge must be straight.
        const auto& grad_cov = edge_grad_cov[edge];
        static constexpr auto kappa = 0.05f;
        using DO::Sara::square;
        const auto cornerness = grad_cov.determinant() -  //
                                kappa * square(grad_cov.trace());
        if (cornerness > 0)
          continue;

        auto rotation = Eigen::Matrix2f{};
        rotation.col(0) = edge_grad_mean[edge].normalized();
        rotation.col(1) = edge_grad_mean[e].normalized();
        const auto d = rotation.determinant();
        if (d > det && d > 0)  // Important check the sign.
        {
          next_edge = e;
          det = d;
        }
      }
      if (det <= 0.5)
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

    // Validate the square.
    auto distinct_vertices = std::unordered_set<int>{};
    for (const auto& v : square)
      distinct_vertices.insert(v);
    if (distinct_vertices.size() != 4)
      return std::nullopt;

#define OLD
#ifdef OLD
    // Validation with the side lengths.
    auto side_lengths = std::array<float, 4>{};
    for (auto i = 0; i < 4; ++i)
    {
      const auto& a = corners[square[i]].coords;
      const auto& b = corners[square[(i + 1) % 4]].coords;
      side_lengths[i] = (b - a).norm();
    }
    const auto [lmin, lmax] =
        std::minmax_element(side_lengths.begin(), side_lengths.end());
    const auto ratio = *lmin / *lmax;
    if (ratio < 0.1f)
      return std::nullopt;
#else
    {
      // Even if very strongly distorted, the putative square should look like a
      // parallelogram.
      const auto& a = corners[square[0]].coords;
      const auto& b = corners[square[1]].coords;
      const auto& c = corners[square[2]].coords;
      const auto& d = corners[square[3]].coords;
      const Eigen::Vector2f ac = c - a;
      const Eigen::Vector2f bd = d - b;

      // calculate the mid-point.
      const Eigen::Vector2f m1 = (a + c) * 0.5f;
      const Eigen::Vector2f m2 = (b + d) * 0.5f;

      const float rel_diff = (m1 - m2).norm() / std::min(ac.norm(), bd.norm());
      if (rel_diff > 0.1f)
        return std::nullopt;
    }

#endif

    return square;
  }

  auto reconstruct_black_square_from_corner(
      const int c, const std::vector<Corner<float>>& corners,
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::optional<std::array<int, 4>>
  {
    auto square = std::optional<std::array<int, 4>>{};
    for (auto d = 0; d < 4; ++d)
    {
      square = reconstruct_black_square_from_corner(
          c, d, corners,  //
          edge_grad_mean, edge_grad_cov, edges_adjacent_to_corner,
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
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
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
      throw std::runtime_error{
          "There must be exactly 4 adjacent edges from the seed corner!"};
    auto edge_it = edges_adjacent_to_corner[square[0]].begin();
    for (auto it = 0; it < start_direction; ++it)
      ++edge_it;
    auto edge = *edge_it;

    // The edge must be straight otherwise it's over.
    {
      const auto& grad_cov = edge_grad_cov[edge];
      static constexpr auto kappa = 0.05f;
      using DO::Sara::square;
      const auto cornerness = grad_cov.determinant() -  //
                              kappa * square(grad_cov.trace());
      if (cornerness > 0)
        return std::nullopt;
    }

    // Pursue the square reconstruction.
    for (auto i = 1; i < 4; ++i)
    {
      square[i] = find_next_square_vertex(edge, square[i - 1],  //
                                          corners,              //
                                          edge_grad_mean,       //
                                          corners_adjacent_to_edge);
      if (square[i] == -1)
        return std::nullopt;

      // The next edge is the one that form the rightmost angle.
      const auto& edges = edges_adjacent_to_corner[square[i]];
      auto next_edge = -1;
      auto det = 1.f;
      for (const auto& e : edges)
      {
        if (e == edge)
          continue;

        // The edge must be straight.
        const auto& grad_cov = edge_grad_cov[edge];
        static constexpr auto kappa = 0.05f;
        using DO::Sara::square;
        const auto cornerness = grad_cov.determinant() -  //
                                kappa * square(grad_cov.trace());
        if (cornerness > 0)
          continue;

        auto rotation = Eigen::Matrix2f{};
        rotation.col(0) = edge_grad_mean[edge].normalized();
        rotation.col(1) = edge_grad_mean[e].normalized();
        const auto d = rotation.determinant();
        if (d < det && d < 0)  // Important check the sign.
        {
          next_edge = e;
          det = d;
        }
      }
      if (det >= -0.5f)
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
      const std::vector<Eigen::Vector2f>& edge_grad_mean,
      const std::vector<Eigen::Matrix2f>& edge_grad_cov,
      const std::vector<std::unordered_set<int>>& edges_adjacent_to_corner,
      const std::vector<std::unordered_set<int>>& corners_adjacent_to_edge)
      -> std::optional<std::array<int, 4>>
  {
    auto square = std::optional<std::array<int, 4>>{};
    for (auto d = 0; d < 4; ++d)
    {
      square = reconstruct_white_square_from_corner(
          c, d, corners,                  //
          edge_grad_mean, edge_grad_cov,  //
          edges_adjacent_to_corner, corners_adjacent_to_edge);
      if (square != std::nullopt)
        return square;
    }
    return std::nullopt;
  }

}  // namespace DO::Sara
