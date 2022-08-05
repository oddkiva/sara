// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include "Corner.hpp"

#include <DO/Sara/Graphics.hpp>

#include <cstdint>
#include <deque>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>


namespace DO::Sara {

  // --------------------------------------------------------------------------
  // ID types.
  // --------------------------------------------------------------------------
  using corner_id_t = std::int32_t;
  using edge_id_t = std::int32_t;
  using edge_t = std::pair<int, int>;
  using square_id_t = std::int32_t;


  // --------------------------------------------------------------------------
  // Containers.
  // --------------------------------------------------------------------------
  struct PairHash final
  {
    template <typename T, typename U>
    inline auto operator()(const std::pair<T, U>& p) const noexcept
        -> std::size_t
    {
      uintmax_t hash = std::hash<T>{}(p.first);
      hash <<= sizeof(uintmax_t) * 4;
      hash ^= std::hash<U>{}(p.second);
      return std::hash<uintmax_t>{}(hash);
    }
  };

  using EdgeIdList = std::unordered_map<edge_t, edge_id_t, PairHash>;
  using EdgesAdjacentToCorner =
      std::unordered_map<corner_id_t, std::unordered_set<edge_id_t>>;
  using SquaresAdjacentToEdge =
      std::unordered_map<edge_id_t, std::unordered_set<square_id_t>>;


  struct Square
  {
    enum class Type : std::uint8_t
    {
      Black,
      White
    };

    std::array<int, 4> v;
    Type type;

    inline auto operator<(const Square& other) const -> bool
    {
      return std::lexicographical_compare(v.begin(), v.end(),  //
                                          other.v.begin(), other.v.end());
    };

    inline auto operator==(const Square& other) const -> bool
    {
      return std::equal(v.begin(), v.end(), other.v.begin());
    }
  };

  using SquareSet = std::set<Square>;


  struct ChessboardSquare
  {
    std::int32_t id;
    Eigen::Vector2i coords;
    std::array<Eigen::Vector2f, 4> dirs;
  };

  using Chessboard = std::deque<std::deque<ChessboardSquare>>;

  inline auto anchor_coords(const Chessboard& cb) -> const Eigen::Vector2i&
  {
    return cb.front().front().coords;
  }

  inline auto rows(const Chessboard& cb) -> int
  {
    return static_cast<int>(cb.size());
  }

  inline auto cols(const Chessboard& cb) -> int
  {
    if (cb.empty())
      return 0;
    return static_cast<int>(cb.front().size());
  }


  // --------------------------------------------------------------------------
  // Print functions
  // --------------------------------------------------------------------------
  inline auto print_square(const std::array<int, 4>& s)
  {
    std::cout << Eigen::Map<const Eigen::RowVector4i>(s.data());
  }

  inline auto print_chessboard_coords(const Chessboard& cb) -> void
  {
    for (const auto& row : cb)
    {
      for (const auto& square : row)
        std::cout << "[" << square.coords.transpose() << "], ";
      std::cout << std::endl;
    }
  }

  inline auto to_matrix(const Chessboard& cb) -> Eigen::MatrixXi
  {
    auto m = Eigen::MatrixXi{cb.size(), cb.front().size()};
    for (auto i = 0; i < m.rows(); ++i)
      for (auto j = 0; j < m.cols(); ++j)
        m(i, j) = cb[i][j].id;
    return m;
  }


  // --------------------------------------------------------------------------
  // Draw functions
  // --------------------------------------------------------------------------
  inline auto draw_square(const std::vector<Corner<float>>& corners,
                          float scale, ImageView<Rgb8>& disp,
                          const std::array<int, 4>& square,  //
                          const Rgb8& color,                 //
                          const int thickness) -> void
  {
#ifdef DEBUG_ME
    static const auto colors = std::array<sara::Rgb8, 4>{
        sara::Red8, sara::Green8, sara::Blue8, sara::Yellow8};
#endif
    for (auto i = 0; i < 4; ++i)
    {
      const Eigen::Vector2f a = corners[square[i]].coords * scale;
      const Eigen::Vector2f b = corners[square[(i + 1) % 4]].coords * scale;
#ifdef DEBUG_ME
      draw_line(disp, a, b, colors[i], thickness);
#else
      draw_line(disp, a, b, color, thickness);
#endif
    }
  };


  // --------------------------------------------------------------------------
  // Utility functions to build the graph of chessboard squares
  // --------------------------------------------------------------------------
  inline auto to_list(const SquareSet& black_squares,
                      const SquareSet& white_squares) -> std::vector<Square>
  {
    auto squares = std::vector<Square>{};
    std::transform(black_squares.begin(), black_squares.end(),
                   std::back_inserter(squares),
                   [](const auto& square) -> Square {
                     // Black square vertices are enumerated in CW order
                     return square;
                   });
    std::transform(
        white_squares.begin(), white_squares.end(), std::back_inserter(squares),
        [](const auto& square) -> Square {
          // White squares are enumerated in CCW order
          // Reversing the order of white square vertices will be
          // necessary to grow chessboards.
          auto square_reversed = square;
          std::reverse(square_reversed.v.begin(), square_reversed.v.end());
          return square_reversed;
        });

    return squares;
  }

  inline auto populate_edge_ids(const std::vector<Square>& squares)
      -> EdgeIdList
  {
    auto edge_ids = EdgeIdList{};

    auto edge_id = 0;
    for (const auto& square : squares)
    {
      for (auto i = 0; i < 4; ++i)
      {
#ifdef DEBUG_EDGE_IDS
        SARA_CHECK(edge_id);
#endif
        const auto& a = square.v[i];
        const auto& b = square.v[(i + 1) % 4];
        if (edge_ids.find({a, b}) == edge_ids.end())
          edge_ids[{a, b}] = edge_id++;
        if (edge_ids.find({b, a}) == edge_ids.end())
          edge_ids[{b, a}] = edge_id++;
      }
    }

#ifdef DEBUG_EDGE_IDS
    for (const auto& [edge, edge_id] : edge_ids)
      SARA_DEBUG << edge_id << " = " << edge.first << " -> " << edge.second
                 << std::endl;
#endif

    return edge_ids;
  }

  inline auto populate_squares_adj_to_edge(const EdgeIdList& edge_ids,
                                           const std::vector<Square>& squares)
      -> SquaresAdjacentToEdge
  {
    auto squares_adj_to_edge = SquaresAdjacentToEdge{};

    for (auto s = 0u; s < squares.size(); ++s)
    {
      const auto s_id = static_cast<int>(s);
      const auto& square = squares[s];
      for (auto i = 0; i < 4; ++i)
      {
        const auto& a = square.v[i];
        const auto& b = square.v[(i + 1) % 4];
        const auto ab = std::make_pair(a, b);
        const auto ba = std::make_pair(b, a);
        const auto e_ab = edge_ids.at(ab);
        const auto e_ba = edge_ids.at(ba);
        squares_adj_to_edge[e_ab].insert(s_id);
        squares_adj_to_edge[e_ba].insert(s_id);
      }
    }

    return squares_adj_to_edge;
  }

  inline auto populate_edges_adj_to_corner(const EdgeIdList& edge_ids,
                                           const std::vector<Square>& squares)
      -> std::pair<EdgesAdjacentToCorner, EdgesAdjacentToCorner>
  {
    auto in_edges = EdgesAdjacentToCorner{};
    auto out_edges = EdgesAdjacentToCorner{};

    for (const auto& square : squares)
    {
      for (auto i = 0; i < 4; ++i)
      {
        const auto& a = square.v[i];
        const auto& b = square.v[(i + 1) % 4];
        const auto e_ab = edge_ids.at({a, b});
        const auto e_ba = edge_ids.at({b, a});
        in_edges[a].insert(e_ba);
        out_edges[a].insert(e_ab);

        in_edges[b].insert(e_ab);
        out_edges[b].insert(e_ba);
      }
    }

    return std::make_pair(in_edges, out_edges);
  }


  // --------------------------------------------------------------------------
  // Region growing
  // --------------------------------------------------------------------------
  inline auto resize_chessboard_if_necessary(Chessboard& cb,
                                             const Eigen::Vector2i& curr_coords,
                                             const Eigen::Vector2i& dir) -> void
  {
    const auto anchor = anchor_coords(cb);
    const Eigen::Vector2i curr_cb_coords = curr_coords - anchor;
    const Eigen::Vector2i next_cb_coords = curr_cb_coords + dir;
    const auto row = next_cb_coords.y();
    const auto col = next_cb_coords.x();

    if (row < 0)
    {
      // Add a new row at the beginning.
      auto new_row = std::deque<ChessboardSquare>{};

      for (auto x = 0; x < cols(cb); ++x)
      {
        new_row.push_back(ChessboardSquare{
            .id = -1,
            .coords = {anchor.x() + x, curr_coords.y() - 1},
            .dirs = {}  //
        });
      }
      cb.emplace_front(std::move(new_row));
    }
    else if (row == static_cast<int>(cb.size()))
    {
      // Add a new row at the end.
      auto new_row = std::deque<ChessboardSquare>{};

      for (auto x = 0; x < cols(cb); ++x)
      {
        new_row.push_back(ChessboardSquare{
            .id = -1,
            .coords = {anchor.x() + x, curr_coords.y() + 1},
            .dirs = {}  //
        });
      }

      cb.emplace_back(std::move(new_row));
    }

    if (col < 0)
    {
      // Add a new column at the beginning.
      for (auto y = 0; y < rows(cb); ++y)
      {
        cb[y].push_front(ChessboardSquare{
            .id = -1,
            .coords = {curr_coords.x() - 1, anchor.y() + y},
            .dirs = {}  //
        });
      }
    }
    else if (col == static_cast<int>(cb.front().size()))
    {
      // Add a new column at the end.
      for (auto y = 0; y < rows(cb); ++y)
      {
        cb[y].push_back(ChessboardSquare{
            .id = -1,
            .coords = {curr_coords.x() + 1, anchor.y() + y},
            .dirs = {}  //
        });
      }
    }

#ifdef INSPECT_REGION_GROWING
    SARA_DEBUG << "CHESSBOARD COORDS" << std::endl;
    print_chessboard_coords(cb);
#endif
  }

  inline auto grow_chessboard(const std::int32_t seed_square_id,
                              const std::vector<Corner<float>>& corners,
                              std::vector<Square>& squares,  //
                              const EdgeIdList& edge_ids,
                              const SquaresAdjacentToEdge& squares_adj_to_edge,
                              std::vector<std::uint8_t>& is_square_visited,
                              [[maybe_unused]] const float scale,  //
                              [[maybe_unused]] ImageView<Rgb8>& display)
      -> Chessboard
  {
    // Initialize the chessboard with seed square.
    auto seed_sq = ChessboardSquare{};
    seed_sq.id = seed_square_id;
    seed_sq.coords = Eigen::Vector2i::Zero();

    const auto& square = squares[seed_square_id].v;
    for (auto i = 0; i < 4; ++i)
    {
      const auto& a = corners[square[i]].coords;
      const auto& b = corners[square[(i + 1) % 4]].coords;
      seed_sq.dirs[i] = (b - a).normalized();
    }

    auto cb = Chessboard{{seed_sq}};

    auto queue = std::queue<ChessboardSquare>{};
    queue.push(seed_sq);

    static const auto dirs = std::array{
        Eigen::Vector2i{0, -1},  // North
        Eigen::Vector2i{1, 0},   // East
        Eigen::Vector2i{0, 1},   // South
        Eigen::Vector2i{-1, 0},  // West
    };

    while (!queue.empty())
    {
      const auto curr_sq = queue.front();
      queue.pop();
      if (is_square_visited[curr_sq.id])
        continue;

      is_square_visited[curr_sq.id] = 1;

      // Look in each cardinal direction.
      const auto& square = squares[curr_sq.id];

#ifdef INSPECT_REGION_GROWING
      SARA_DEBUG << "SQUARE: " << curr_sq.id << "\n"
                 << Eigen::Map<const Eigen::RowVector4i>(square.v.data())
                 << std::endl;
      SARA_DEBUG << "CHESSBOARD:\n" << to_matrix(cb) << std::endl;
      draw_square(corners, scale, display, square.v, sara::Green8, 3);
      sara::display(display);
      sara::get_key();
#endif

      for (auto i = 0; i < 4; ++i)
      {
#ifdef DEBUG_REGION_GROWING
        std::cout << "\nSEARCH DIRECTION " << i << std::endl;
        SARA_DEBUG << "DIR VECTOR     = " << dirs[i].transpose() << std::endl;
#endif
        const auto& a = square.v[i];
        const auto& b = square.v[(i + 1) % 4];
        const auto& ab = edge_ids.at({a, b});

        const auto& adj_squares = squares_adj_to_edge.at(ab);
        if (adj_squares.empty())
          throw std::runtime_error{
              "The current square must be in the list of squares "
              "adjacent to its own edge!"};

        // Remove the current square from the list of adjacent squares.
        auto neighbor_squares = adj_squares;
        neighbor_squares.erase(curr_sq.id);
        if (neighbor_squares.size() > 1)
        {
          // TODO: choose the biggest square that contains everything,
          // there must be no overlap area.
          std::cerr << "TODO: address ambiguity later..." << std::endl;
          continue;
        }

        if (neighbor_squares.empty())
          continue;

        const auto& neighbor_square_id = *neighbor_squares.begin();

        // Reallocate the chessboard.
        resize_chessboard_if_necessary(cb, curr_sq.coords, dirs[i]);

        // Retrieve the allocated square in the chessboard.
        const Eigen::Vector2i neighbor_pos = curr_sq.coords -     //
                                             anchor_coords(cb) +  //
                                             dirs[i];

#ifdef DEBUG_REGION_GROWING
        SARA_DEBUG << "NEIGHBOR ID = " << neighbor_square_id << std::endl;
        SARA_DEBUG << "CB ANCHOR COORDS = " << anchor_coords(cb).transpose()
                   << std::endl;
        SARA_DEBUG << "CB NEIGHB COORDS = " << neighbor_pos.transpose()
                   << std::endl;
#endif

        // Retrieve the corresponding square in the chessboard.
        auto& allocated_square = cb[neighbor_pos.y()][neighbor_pos.x()];
        // Update the chessboard with the newsquare.
        if (allocated_square.id == -1)
          allocated_square.id = neighbor_square_id;
        // There is already a square assigned.
        else if (allocated_square.id != neighbor_square_id)
          std::cerr
              << "TODO: choose the best neighbor but going ahead anyways..."
              << std::endl;

        // Change the order of the neighbor square vertices so that its sides
        // are enumerated in the same order.
        auto& neighbor_vertices = squares[neighbor_square_id].v;
#ifdef DEBUG_REGION_GROWING
        SARA_DEBUG << "NEIGHBOR SQUARE: "
                   << Eigen::Map<const Eigen::RowVector4i>(
                          neighbor_vertices.data())
                   << std::endl;
#endif

        const auto curr_dirs = curr_sq.dirs;
        auto neighbor_dirs = std::array<Eigen::Vector2f, 4>{};
        for (auto i = 0; i < 4; ++i)
        {
          std::rotate(neighbor_vertices.begin(), neighbor_vertices.begin() + 1,
                      neighbor_vertices.end());

          for (auto k = 0; k < 4; ++k)
          {
            const auto& a = corners[neighbor_vertices[k]].coords;
            const auto& b = corners[neighbor_vertices[(k + 1) % 4]].coords;
            neighbor_dirs[k] = (b - a).normalized();
          }

          auto dots = std::array<float, 4>{};
          std::transform(curr_dirs.begin(), curr_dirs.end(),
                         neighbor_dirs.begin(), dots.begin(),
                         [](const Eigen::Vector2f& a,
                            const Eigen::Vector2f& b) { return a.dot(b); });

#ifdef DEBUG_SQUARE_ROTATION
          std::cout << "Rotation " << i << std::endl;
          print_square(neighbor_vertices);
          std::cout << std::endl;
          for (auto k = 0; k < 4; ++k)
            std::cout << "curr[" << k << "] = " << curr_dirs[k].transpose()
                      << std::endl;
          for (auto k = 0; k < 4; ++k)
            std::cout << "neig[" << k << "] = " << neighbor_dirs[k].transpose()
                      << std::endl;
          std::cout << "dots = "
                    << Eigen::Map<const Eigen::RowVector4f>(dots.data())
                    << std::endl;
#endif

          if (std::all_of(dots.begin(), dots.end(),
                          [](const auto& dot) { return dot > 0.8f; }))
            break;
        }

#ifdef DEBUG_SQUARE_ROTATION
        SARA_DEBUG << "PARENT DIR:\n";
        for (auto i = 0; i < 4; ++i)
          std::cout << "[" << i << "] " << curr_dirs[i].transpose()
                    << std::endl;
        SARA_DEBUG << "NEIGHB DIR:\n";
        for (auto i = 0; i < 4; ++i)
          std::cout << "[" << i << "] " << neighbor_dirs[i].transpose()
                    << std::endl;
#endif
        // Save the new direction.
        allocated_square.dirs = neighbor_dirs;

#ifdef INSPECT_REGION_GROWING
        SARA_DEBUG << "CHESSBOARD STATE:\n" << to_matrix(cb) << std::endl;
        draw_square(corners, scale, display, neighbor_vertices, sara::Magenta8,
                    3);
#endif

        if (!is_square_visited[neighbor_square_id])
          queue.push(allocated_square);
      }

#ifdef INSPECT_REGION_GROWING
      display(display);
      get_key();
#endif
    }

    return cb;
  }

}  // namespace DO::Sara
