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

//! @example

#include <omp.h>

#include <unordered_map>
#include <unordered_set>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>

#include "Chessboard/ChessboardDetector.hpp"
#include "Utilities/ImageOrVideoReader.hpp"


namespace sara = DO::Sara;


struct PairHash final
{
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U>& p) const noexcept
  {
    uintmax_t hash = std::hash<T>{}(p.first);
    hash <<= sizeof(uintmax_t) * 4;
    hash ^= std::hash<U>{}(p.second);
    return std::hash<uintmax_t>{}(hash);
  }
};

using corner_id_t = std::int32_t;
using edge_id_t = std::int32_t;
using edge_t = std::pair<int, int>;
using square_id_t = std::int32_t;
using EdgeIdList = std::map<edge_t, edge_id_t>;
using SquareSet = sara::ChessboardDetector::SquareSet;
using SquaresAdjacentToEdge =
    std::unordered_map<edge_id_t, std::unordered_set<square_id_t>>;
using EdgesAdjacentToCorner =
    std::unordered_map<corner_id_t, std::unordered_set<edge_id_t>>;

auto draw_square(const std::vector<Corner<float>>& corners, float scale,
                 sara::ImageView<sara::Rgb8>& disp,
                 const std::array<int, 4>& square,  //
                 const sara::Rgb8& color,           //
                 const int thickness) -> void
{
  for (auto i = 0; i < 4; ++i)
  {
    const Eigen::Vector2f a = corners[square[i]].coords * scale;
    const Eigen::Vector2f b = corners[square[(i + 1) % 4]].coords * scale;
    sara::draw_line(disp, a, b, color, thickness);
  }
};

struct ChessboardSquare
{
  std::int32_t id;
  Eigen::Vector2i pos;
  std::array<Eigen::Vector2f, 4> dirs;
};
using Chessboard = std::deque<std::deque<ChessboardSquare>>;

struct Square
{
  enum class Type : std::uint8_t
  {
    Black,
    White
  };

  std::array<int, 4> v;
  Type type;
};


auto to_list(const SquareSet& black_squares, const SquareSet& white_squares)
    -> std::vector<Square>
{
  auto squares = std::vector<Square>{};
  std::transform(black_squares.begin(), black_squares.end(),
                 std::back_inserter(squares), [](const auto& square) -> Square {
                   return {square, Square::Type::Black};
                 });
  std::transform(white_squares.begin(), white_squares.end(),
                 std::back_inserter(squares), [](const auto& square) -> Square {
                   return {square, Square::Type::White};
                 });

  return squares;
}

auto to_matrix(const Chessboard& cb) -> Eigen::MatrixXi
{
  auto m = Eigen::MatrixXi{cb.size(), cb.front().size()};
  for (auto i = 0; i < m.rows(); ++i)
    for (auto j = 0; j < m.cols(); ++j)
      m(i, j) = cb[i][j].id;
  return m;
}

auto populate_edge_ids(const std::vector<Square>& squares) -> EdgeIdList
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

auto populate_squares_adj_to_edge(const EdgeIdList& edge_ids,
                                  const std::vector<Square>& squares)
    -> SquaresAdjacentToEdge
{
  auto squares_adj_to_edge = SquaresAdjacentToEdge{};

  SARA_CHECK(squares.size());
  for (auto s = 0u; s < squares.size(); ++s)
  {
    const auto s_id = static_cast<int>(s);
    const auto& square = squares[s];
    SARA_DEBUG << "square " << s_id << " -> "
               << Eigen::Map<const Eigen::RowVector4i>(square.v.data())
               << std::endl;
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

auto populate_edges_adj_to_corner(const EdgeIdList& edge_ids,
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

auto resize_chessboard_if_necessary(Chessboard& cb,
                                    const Eigen::Vector2i& curr_pos,
                                    const Eigen::Vector2i& dir) -> void
{
  SARA_DEBUG << "RESIZING CHESSBOARD" << std::endl;
  const auto& anchor_pos = cb.front().front().pos;
  const Eigen::Vector2i curr_pos_in_cb = curr_pos - anchor_pos;
  const Eigen::Vector2i next_pos_in_cb = curr_pos_in_cb + dir;
  SARA_DEBUG << "- CURR = " << curr_pos_in_cb.transpose() << std::endl;
  SARA_DEBUG << "- DIR  = " << dir.transpose() << std::endl;
  SARA_DEBUG << "- NEXT = " << next_pos_in_cb.transpose() << std::endl;
  const auto row = next_pos_in_cb.y();
  const auto col = next_pos_in_cb.x();

  if (row < 0)
  {
    // Add a new row at the beginning.
    auto new_row = std::deque<ChessboardSquare>{};
    for (auto x = 0; x <= curr_pos_in_cb.x(); ++x)
    {
      new_row.push_back(ChessboardSquare{
          .id = -1,                                                      //
          .pos = Eigen::Vector2i{anchor_pos.x() + x, curr_pos.y() - 1},  //
          .dirs = {}                                                     //
      });
    }
    cb.emplace_front(std::move(new_row));
  }
  else if (row == static_cast<int>(cb.size()))
  {
    // Add a new row at the end.
    auto new_row = std::deque<ChessboardSquare>{};
    for (auto x = 0; x <= curr_pos_in_cb.x(); ++x)
    {
      new_row.push_back(ChessboardSquare{
          .id = -1,                                                      //
          .pos = Eigen::Vector2i{anchor_pos.x() + x, curr_pos.y() + 1},  //
          .dirs = {}                                                     //
      });
    }
    cb.emplace_back(std::move(new_row));
  }
  if (col < 0)
  {
    // Add a new column at the beginning.
    for (auto y = 0u; y < cb.size(); ++y)
    {
      cb[y].push_front(ChessboardSquare{
          .id = -1,
          .pos = Eigen::Vector2i{curr_pos_in_cb.x() - 1, anchor_pos.y() + y},
          .dirs = {}  //
      });
    }
  }
  else if (col == static_cast<int>(cb.front().size()))
  {
    // Add a new column at the end.
    for (auto y = 0u; y < cb.size(); ++y)
    {
      cb[y].push_back(ChessboardSquare{
          .id = -1,
          .pos = Eigen::Vector2i{curr_pos_in_cb.x() + 1, anchor_pos.y() + y},
          .dirs = {}  //
      });
    }
  }
}

auto grow_chessboard(const std::int32_t seed_square_id,
                     const std::vector<Corner<float>>& corners,
                     std::vector<Square>& squares,  //
                     const EdgeIdList& edge_ids,
                     const SquaresAdjacentToEdge& squares_adj_to_edge,
                     std::vector<std::uint8_t>& is_square_visited,
                     const float scale, sara::ImageView<sara::Rgb8>& display)
    -> Chessboard
{
  // Initialize the chessboard with seed square.
  auto seed_sq = ChessboardSquare{};
  seed_sq.id = seed_square_id;
  seed_sq.pos = Eigen::Vector2i::Zero();

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

#define INSPECT_REGION_GROWING
#ifdef INSPECT_REGION_GROWING
    SARA_DEBUG << "SQUARE: " << curr_sq.id << "\n"
               << Eigen::Map<const Eigen::RowVector4i>(square.v.data())
               << std::endl;
    SARA_DEBUG << "CHESSBOARD:\n" << to_matrix(cb) << std::endl;
    draw_square(corners, scale, display, square.v, sara::Green8, 3);
    sara::display(display);
#endif

    for (auto i = 0; i < 4; ++i)
    {
      SARA_DEBUG << "SEARCH DIRECTION " << i << std::endl;
      SARA_DEBUG << "DIR VECTOR     = " << dirs[i].transpose() << std::endl;
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
      SARA_DEBUG << "Neighbor squares: ";
      for (const auto& square : neighbor_squares)
        std::cout << square << ", ";
      std::cout << std::endl;

      if (neighbor_squares.empty())
        continue;

      const auto& neighbor_square_id = *neighbor_squares.begin();
      SARA_DEBUG << "NEIGHBOR ID = " << neighbor_square_id << std::endl;

      // Reallocate the chessboard.
      resize_chessboard_if_necessary(cb, curr_sq.pos, dirs[i]);
      SARA_DEBUG << "[BEFORE] CHESSBOARD:\n" << to_matrix(cb) << std::endl;
      SARA_DEBUG << "CB ANCHOR COORDS = " << cb.front().front().pos.transpose()
                 << std::endl;

      // Retrieve the allocated square in the chessboard.
      const Eigen::Vector2i neighbor_pos = curr_sq.pos -             //
                                           cb.front().front().pos +  //
                                           dirs[i];
      SARA_DEBUG << "CB NEIGHB COORDS = " << neighbor_pos.transpose()
                 << std::endl;


      // Retrieve the corresponding square in the chessboard.
      auto& allocated_square = cb[neighbor_pos.y()][neighbor_pos.x()];
      // Update the chessboard with the newsquare.
      if (allocated_square.id == -1)
        allocated_square.id = neighbor_square_id;
      // There is already a square assigned.
      else if (allocated_square.id != neighbor_square_id)
        std::cerr << "TODO: choose the best neighbor..." << std::endl;

      // Change the order of the neighbor square vertices so that its sides are
      // enumerated in the same order.
      auto& neighbor_vertices = squares[neighbor_square_id].v;
      SARA_DEBUG << "NEIGHBOR SQUARE: "
                 << Eigen::Map<const Eigen::RowVector4i>(
                        neighbor_vertices.data())
                 << std::endl;

      const auto curr_dirs = curr_sq.dirs;
      auto neighbor_dirs = std::array<Eigen::Vector2f, 4>{};
      for (auto i = 0; i < 4; ++i)
      {
        std::rotate(neighbor_vertices.begin(), neighbor_vertices.begin() + 1,
                    neighbor_vertices.end());
        for (auto k = 0; k < 4; ++k)
        {
          const auto& a = corners[neighbor_vertices[i]].coords;
          const auto& b = corners[neighbor_vertices[(i + 1) % 4]].coords;
          neighbor_dirs[k] = (b - a).normalized();
        }

        auto dots = std::array<float, 4>{};
        std::transform(curr_dirs.begin(), curr_dirs.end(),
                       neighbor_dirs.begin(), dots.begin(),
                       [](const Eigen::Vector2f& a, const Eigen::Vector2f& b) {
                         return a.dot(b);
                       });
        if (std::all_of(dots.begin(), dots.end(),
                        [](const auto& dot) { return dot > 0.8f; }))
          break;
      }
      // Save the new direction.
      allocated_square.dirs = neighbor_dirs;

#ifdef INSPECT_REGION_GROWING
      SARA_DEBUG << "[AFTER ] CHESSBOARD:\n" << to_matrix(cb) << std::endl;
      draw_square(corners, scale, display, neighbor_vertices, sara::Magenta8,
                  3);
      sara::display(display);
      sara::get_key();
#endif

      if (!is_square_visited[neighbor_square_id])
        queue.push(allocated_square);
    }
  }

  return cb;
}

auto __main(int argc, char** argv) -> int
{
  try
  {
    omp_set_num_threads(omp_get_max_threads());

#ifdef _WIN32
    const auto video_file = sara::select_video_file_from_dialog_box();
    if (video_file.empty())
      return 1;
#else
    if (argc < 2)
      return 1;
    const auto video_file = std::string{argv[1]};
#endif

    // Visual inspection option
    const auto pause = argc < 3 ? false : static_cast<bool>(std::stoi(argv[2]));
    const auto check_edge_map = argc < 4
                                    ? false  //
                                    : static_cast<bool>(std::stoi(argv[3]));

    // Setup the detection parameters.
    auto params = sara::ChessboardDetector::Parameters{};
    if (argc >= 5)
      params.downscale_factor = std::stof(argv[4]);
    if (argc >= 6)
      params.cornerness_adaptive_thres = std::stof(argv[5]);
    if (argc >= 7)
    {
      const auto value = std::stoi(argv[6]);
      if (value != -1)
        params.corner_filtering_radius = value;
      else
        params.set_corner_nms_radius();
    }
    else
      params.set_corner_nms_radius();
    if (argc >= 8)
    {
      const auto value = std::stoi(argv[7]);
      if (value != -1)
        params.corner_edge_linking_radius = value;
      else
        params.set_corner_edge_linking_radius_to_corner_filtering_radius();
    }
    else
      params.set_corner_edge_linking_radius_to_corner_filtering_radius();


    auto timer = sara::Timer{};
    auto video_stream = sara::ImageOrVideoReader{video_file};
    auto video_frame = video_stream.frame();
    auto frame_gray = sara::Image<float>{video_frame.sizes()};
    auto display = sara::Image<sara::Rgb8>{video_frame.sizes()};
    auto frame_number = -1;

    auto detect = sara::ChessboardDetector{params};

    while (video_stream.read())
    {
      ++frame_number;
      if (frame_number % 3 != 0)
        continue;
      SARA_DEBUG << "Frame #" << frame_number << std::endl;

      if (sara::active_window() == nullptr)
      {
        sara::create_window(video_frame.sizes(), video_file);
        sara::set_antialiasing();
      }

      timer.restart();
      {
        sara::tic();
        sara::from_rgb8_to_gray32f(video_frame, frame_gray);
        sara::toc("Grayscale conversion");
        detect(frame_gray);
      }
      const auto pipeline_time = timer.elapsed_ms();
      SARA_DEBUG << "Processing time = " << pipeline_time << "ms" << std::endl;

      sara::tic();
      if (check_edge_map)
      {
        // Resize
        auto display_32f_ds = detect._ed.pipeline.edge_map.convert<float>();
        auto display_32f = sara::Image<float>{video_frame.sizes()};
        sara::scale(display_32f_ds, display_32f);

        display = display_32f.convert<sara::Rgb8>();
      }
      else
        display = frame_gray.convert<sara::Rgb8>();

      const auto num_corners = static_cast<int>(detect._corners.size());

      const auto& radius = detect._params.corner_filtering_radius;
      const auto& scale = detect._params.downscale_factor;
      const auto draw_corner = [radius,
                                scale](sara::ImageView<sara::Rgb8>& display,
                                       const Corner<float>& c,
                                       const sara::Rgb8& color, int thickness) {
        const Eigen::Vector2i p1 =
            (scale * c.coords).array().round().cast<int>();
        sara::fill_circle(display, p1.x(), p1.y(), 1, sara::Yellow8);
        sara::draw_circle(display, p1.x(), p1.y(),
                          static_cast<int>(std::round(radius * scale)), color,
                          thickness);
      };

#pragma omp parallel for
      for (auto c = 0; c < num_corners; ++c)
      {
        const auto good = sara::is_seed_corner(   //
            detect._edges_adjacent_to_corner[c],  //
            detect._gradient_peaks_refined[c],    //
            detect._zero_crossings[c],            //
            detect.N);

        // Remove noisy corners to understand better the behaviour of the
        // algorithm.
        if (detect._edges_adjacent_to_corner[c].empty())
          continue;

        const auto& corner = detect._corners[c];
        draw_corner(display, corner, good ? sara::Red8 : sara::Cyan8, 2);
      }
      sara::draw_text(display, 80, 80, std::to_string(frame_number),
                      sara::White8, 60, 0, false, true);

      const auto& corners = detect._corners;
#ifdef SHOW_LINES
      const auto& lines = detect._lines;
      for (const auto& line : lines)
      {
        for (auto i = 0u; i < line.size() - 1; ++i)
        {
          const auto& ca = corners[line[i]];
          const auto& cb = corners[line[i + 1]];
          const Eigen::Vector2f a = ca.coords * scale;
          const Eigen::Vector2f b = cb.coords * scale;
          sara::draw_line(display, a, b, sara::Cyan8, 2);
          draw_corner(display, ca, sara::Green8, 2);
          draw_corner(display, cb, sara::Green8, 2);
        }
      }
#endif

#define SHOW_SQUARES
#ifdef SHOW_SQUARES
      for (const auto& square : detect._white_squares)
        draw_square(corners, scale, display, square, sara::Red8, 3);
      for (const auto& square : detect._black_squares)
        draw_square(corners, scale, display, square, sara::Green8, 2);
#endif

      sara::display(display);
      sara::millisleep(20);

      const auto& black_squares = detect._black_squares;
      const auto& white_squares = detect._white_squares;

      auto squares = to_list(black_squares, white_squares);

      // Populate edge IDs.
      const auto edge_ids = populate_edge_ids(squares);
      const auto squares_adj_to_edge =
          populate_squares_adj_to_edge(edge_ids, squares);
      const auto [in_edges, out_edges] =
          populate_edges_adj_to_corner(edge_ids, squares);

      SARA_CHECK(edge_ids.size());
      SARA_CHECK(squares_adj_to_edge.size());
      SARA_CHECK(in_edges.size());
      SARA_CHECK(out_edges.size());

      auto is_square_visited = std::vector<std::uint8_t>(squares.size(), 0);
      auto square_ids = std::queue<int>{};
      for (auto s = 0u; s < squares.size(); ++s)
        square_ids.push(s);

      // Build chessboards.
      auto chessboards = std::vector<Chessboard>{};
      while (!square_ids.empty())
      {
        // The seed square.
        const auto square_id = square_ids.front();
        square_ids.pop();
        if (is_square_visited[square_id])
          continue;

        SARA_DEBUG << "SEED SQUARE = " << square_id << std::endl;

        auto display = frame_gray.convert<sara::Rgb8>();
        sara::display(display);
        grow_chessboard(square_id, corners, squares, edge_ids,
                        squares_adj_to_edge, is_square_visited,  //
                        scale, display);
      }

      if (pause)
        sara::get_key();
    }
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  return 0;
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}
