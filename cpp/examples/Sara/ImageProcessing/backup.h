

#define VP
#ifdef VP
    tic();
    // Statistical line estimation.
    const auto& line_segments = extract_line_segments_quick_and_dirty(  //
        curve_matcher.stats_curr                                        //
    );
    const auto& lines = to_lines(line_segments);

    const auto [vph, inliers, best_line_pair] =
        find_horizontal_vanishing_point(lines, 2.5, 100);
    SARA_CHECK(vph.transpose());
    SARA_CHECK(inliers.size());
    toc("VP Detection");
#endif

#ifdef VP
    // Draw the line segments.
    SARA_CHECK(line_segments.size());
    for (const auto& ls : line_segments)
    {
      const auto p1 = (ls.p1() * double(s) + p1d).cast<int>();
      const auto p2 = (ls.p2() * double(s) + p1d).cast<int>();
      draw_line(detection, p1.x(), p1.y(), p2.x(), p2.y(), Blue8, 2);
    }

    // Show the vanishing point.
    const Eigen::Vector2f vp = vph.hnormalized() * s + p1d.cast<float>();
    fill_circle(detection, vp.x(), vp.y(), 5, Magenta8);
#endif



struct CurveMatcher
{
  std::vector<std::vector<Eigen::Vector2d>> curves_prev;
  std::vector<std::vector<Eigen::Vector2d>> curves_curr;

  CurveStatistics stats_prev;
  CurveStatistics stats_curr;

  Image<int> curve_map_prev;
  Image<int> curve_map_curr;

  auto reset_curve_map(int w, int h) -> void
  {
    curve_map_prev.resize({w, h});
    curve_map_curr.resize({w, h});

    curve_map_prev.flat_array().fill(-1);
    curve_map_curr.flat_array().fill(-1);
  }

  auto recalculate_curve_map(
      const std::vector<std::vector<Eigen::Vector2i>>& curve_points) -> void
  {
    curve_map_curr.flat_array().fill(-1);
#pragma omp parallel for
    for (auto i = 0; i < static_cast<int>(curve_points.size()); ++i)
    {
      const auto& points = curve_points[i];
      for (const auto& p : points)
        curve_map_curr(p) = i;
    }
  }

  auto update_curve_features(
      const std::vector<std::vector<Eigen::Vector2d>>& curves_as_polylines)
  {
    curves_curr.swap(curves_prev);
    curves_curr = curves_as_polylines;

    stats_curr.swap(stats_prev);
    stats_curr = CurveStatistics(curves_curr);

    curve_map_curr.swap(curve_map_prev);
    // recalculate_curve_map(curve_points);
  }
};



+  auto curve_matcher = CurveMatcher{};
+  curve_matcher.reset_curve_map(frame.width(), frame.height());
+

+    // Perform curve matching.
+    curve_matcher.update_curve_features(edges);
+

-    for (const auto& e : edges_refined)
-      if (e.size() >= 2)
-        draw_polyline(detection, e, Blue8, p1d, s);
+// #define INSPECT_CURVES_FOR_MATCHING
+#ifdef INSPECT_CURVES_FOR_MATCHING
+    for (auto i = 0u; i < curve_matcher.curves_prev.size(); ++i)
+    {
+      const auto& e = curve_matcher.curves_prev[i];
+      // curve_matcher.stats_prev.oriented_box(i).draw(detection, Red8, p1d,
+      // s);
+      draw_polyline(detection, e, Magenta8, p1d, s);
+    }
