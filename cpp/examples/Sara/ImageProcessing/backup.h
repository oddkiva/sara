struct EdgeGraph
{
  const EdgeAttributes& attributes;
  std::vector<std::size_t> edge_ids;

  Tensor_<std::uint8_t, 2> A;

  EdgeGraph(const EdgeAttributes& attrs)
    : attributes{attrs}
  {
    initialize();
  }

  auto initialize() -> void
  {
    const auto n = static_cast<std::int32_t>(attributes.edges.size());
    std::iota(edge_ids.begin(), edge_ids.end(), 0u);

    A.resize(n, n);
    A.flat_array().fill(0);
    for (auto i = 0; i < n; ++i)
    {
      for (auto j = i; j < n; ++j)
      {
        if (i == j)
          A(i, j) = 1;
        else if (edge(i).size() >= 2 && edge(j).size() >= 2)
          A(i, j) = is_aligned(i, j);
        A(j, i) = A(i, j);
      }
    }
  }

  auto edge(std::size_t i) const -> const Edge&
  {
    return attributes.edges[i];
  }

  auto center(std::size_t i) const -> const Vector2d&
  {
    return attributes.centers[i];
  }

  auto axes(std::size_t i) const -> const Matrix2d&
  {
    return attributes.axes[i];
  }

  auto lengths(std::size_t i) const -> const Vector2d&
  {
    return attributes.lengths[i];
  }

  auto rect(std::size_t i) const -> Rectangle
  {
    return {center(i), axes(i), lengths(i)};
  }

  auto is_aligned(std::size_t i, std::size_t j) const -> bool
  {
    const auto& ri = rect(i);
    const auto& rj = rect(j);

    const auto& di = ri.axes.col(0);
    const auto& dj = rj.axes.col(0);

    const auto& ci = ri.center.homogeneous().eval();
    const auto& cj = rj.center.homogeneous().eval();

    const auto& li = ri.line().normalized();
    const auto& lj = rj.line().normalized();

    return Projective::point_to_line_distance(ci, lj) < 5. &&
           Projective::point_to_line_distance(cj, li) < 5. &&
           std::abs(di.dot(dj)) > std::cos(10._deg);
  }

  auto group_by_alignment() const
  {
    const auto n = A.rows();

    auto ds = DisjointSets(n);

    for (auto i = 0; i < n; ++i)
      ds.make_set(i);

    for (auto i = 0; i < n; ++i)
      for (auto j = i; j < n; ++j)
        if (A(i, j) == 1)
          ds.join(ds.node(i), ds.node(j));

    auto groups = std::map<std::size_t, std::vector<std::size_t>>{};
    for (auto i = 0; i < n; ++i)
    {
      const auto c = ds.component(i);
      groups[c].push_back(i);
    }

    return groups;
  }
};


auto check_edge_grouping(const ImageView<Rgb8>& frame,                 //
                         const std::vector<Edge>& edges_refined,       //
                         const std::vector<Eigen::Vector2d>& centers,  //
                         const std::vector<Eigen::Matrix2d>& axes,     //
                         const std::vector<Eigen::Vector2d>& lengths,  //
                         const Point2i& p1,                            //
                         double downscale_factor)                      //
    -> void
{
  tic();
  const auto edge_attributes = EdgeAttributes{.edges = edges_refined,
                                              .centers = centers,
                                              .axes = axes,
                                              .lengths = lengths};
  const auto edge_graph = EdgeGraph{edge_attributes};
  const auto edge_groups = edge_graph.group_by_alignment();
  SARA_CHECK(edge_groups.size());
  toc("Edge Grouping By Alignment");


  // Display the quasi-straight edges.
  auto draw_task = [=]() {
    auto edge_group_colors = std::map<std::size_t, Rgb8>{};
    for (const auto& g : edge_groups)
      edge_group_colors[g.first] << rand() % 255, rand() % 255, rand() % 255;

    auto edge_colors = std::vector<Rgb8>(edges_refined.size(), Red8);
    // for (auto& c : edge_colors)
    //   c << rand() % 255, rand() % 255, rand() % 255;
    for (const auto& g : edge_groups)
      for (const auto& e : g.second)
        edge_colors[e] = edge_group_colors[g.first];

    const Eigen::Vector2d p1d = p1.cast<double>();
    const auto& s = downscale_factor;

    auto detection = Image<Rgb8>{frame};
    detection.flat_array().fill(Black8);
    for (const auto& g : edge_groups)
    {
      for (const auto& e : g.second)
      {
        // const auto& edge = edges_simplified[e];
        const auto& edge_refined = edges_refined[e];
        if (edge_refined.size() < 2)
          continue;

        if (length(edge_refined) < 5)
          continue;

        const auto& theta = std::abs(std::atan2(axes[e](1, 0), axes[e](0, 0)));
        if (theta < 10._deg || std::abs(M_PI - theta) < 10._deg)
          continue;

        const auto& color = edge_colors[e];
        draw_polyline(detection, edge_refined, color, p1d, s);

        const Point2d& c = center_of_mass(edge_refined);
        const Point2d& c1 = p1d + s * c;
        fill_circle(detection, c1.x(), c1.y(), 3, color);

// #define DEBUG_SHAPE_STATISTICS
#ifdef DEBUG_SHAPE_STATISTICS
        const auto& rect = Rectangle{.center = c,      //
                                     .axes = axes[e],  //
                                     .lengths = lengths[e]};
        rect.draw(detection, White8, p1d, s);
#endif
      }
    }

    display(detection);
  };

  tic();
  // std::async(std::launch::async, draw_task);
  draw_task();
  toc("Draw");
}


