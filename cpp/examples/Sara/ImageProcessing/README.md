# Chessboard Detection

- Hessian Detection: very good idea but turns out to be quite tricky to filter
  and set the appropriate threshold for every case.

- Otsu binarization method works OK, but too unstable w.r.t. illumination
  changes.

  - See the implementation in `drafts/ImageProcessing/Otsu.hpp`.

- Color-based grouping via watershed works, but unstable w.r.t. illumination
  changes

  ```
  auto mean_colors(const std::map<int, std::vector<Eigen::Vector2i>>& regions,
                   const sara::Image<sara::Rgb8>& image)
  {
    auto colors = std::map<int, sara::Rgb8>{};
    for (const auto& [label, points] : regions)
    {
      const auto num_points = points.size();
      Eigen::Vector3f color = Eigen::Vector3f::Zero();
      for (const auto& p : points)
        color += image(p).cast<float>();
      color /= num_points;

      colors[label] = color.cast<std::uint8_t>();
    }
    return colors;
  }


  // Watershed to find the chessboard quadrangles.
  const auto color_threshold = std::sqrt(std::pow(2, 2) * 3);
  const auto segment_min_size = 50;

  const auto regions = sara::color_watershed(image, color_threshold);
  const auto colors = mean_colors(regions, image);
  auto partitioning = sara::Image<sara::Rgb8>{image.sizes()};
  for (const auto& [label, points] : regions)
  {
    // Show big segments only.
    for (const auto& p : points)
      partitioning(p) = points.size() < segment_min_size  //
                            ? sara::Black8
                            : colors.at(label);
  }
  sara::display(partitioning);

  for (const auto& p : saddle_points)
    sara::draw_circle(p.p, 5, sara::Green8, 2);
  ```

- The approach is to use a hierarchical nedge-grouping approach to circumvent
  illumination changes.

  - A chessboard corner is easily interpretable by means of gradient analysis.
    For any line that divides each chessboard square piece, the gradient direction
    is the same, but its gradient flips at each corner.




- Connecting end points of edges.

  ```
  auto endpoint_graph = sara::EndPointGraph{edge_attributes};
  endpoint_graph.mark_plausible_alignments();
  sara::toc("Alignment Computation");

  // Draw alignment-based connections.
  const auto& score = endpoint_graph.score;
  for (auto i = 0; i < score.rows(); ++i)
  {
    for (auto j = i + 1; j < score.cols(); ++j)
    {
      const auto& pi = endpoint_graph.endpoints[i];
      const auto& pj = endpoint_graph.endpoints[j];

      if (score(i, j) != std::numeric_limits<double>::infinity())
      {
        sara::draw_line(image, pi.x(), pi.y(), pj.x(), pj.y(), sara::Yellow8,
                        2);
        sara::draw_circle(image, pi.x(), pi.y(), 3, sara::Yellow8, 3);
        sara::draw_circle(image, pj.x(), pj.y(), 3, sara::Yellow8, 3);
      }
    }
  }

  const auto edge_groups = endpoint_graph.group();
  auto edge_group_colors = std::map<std::size_t, sara::Rgb8>{};
  for (const auto& g : edge_groups)
    edge_group_colors[g.first] << rand() % 255, rand() % 255, rand() % 255;

  for (const auto& g : edge_groups)
  {
    for (const auto& e : g.second)
    {
      const auto& edge = edges_simplified[e];
      const auto& color = edge_group_colors[e];
      for (const auto& p : edge)
        sara::fill_circle(p.x(), p.y(), 2, color);
    }
  }
  ```
