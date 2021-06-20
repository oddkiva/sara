# Chessboard Detection

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
