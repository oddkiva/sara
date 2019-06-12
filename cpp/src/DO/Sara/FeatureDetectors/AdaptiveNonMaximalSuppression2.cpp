//auto non_max_suppression(const MatrixXf& boxes,
//                            const VectorXf& scores,
//                            float threshold) -> std::deque<int>
//   {
//     auto y1 = boxes.col(0);
//     auto x1 = boxes.col(1);
//     auto y2 = boxes.col(2);
//     auto x2 = boxes.col(3);
//
//     const VectorXf area = ((y2 - y1).array() * (x2 - x1).array());
//
//     const VectorXi sorted_boxes_array = sorted(scores);
//
//     auto sorted_boxes =
//        std::deque<int>{sorted_boxes_array.data(),
//                        sorted_boxes_array.data() + sorted_boxes_array.size()};
//     auto filtered_boxes = std::deque<int>{};
//
//     auto overlaps = [threshold](float val) {
//        return val > threshold;
//     };
//
//     while (!sorted_boxes.empty())
//     {
//       // The current best box for which we need to remove redundant boxes.
//       const auto i_best = sorted_boxes.front();
//
//       // Save it.
//       filtered_boxes.push_back(i_best);
//
//       // Remove it from the garbage list.
//       sorted_boxes.pop_front();
//
//       // Get the list of intersection over union ratios.
//       const auto iou =
//          compute_iou(boxes.row(i_best), row_slice(boxes, sorted_boxes),
//                      area[i_best], row_slice(area, sorted_boxes));
//
//       // The list of indices of boxes overlapping with the current best box.
//       const auto redundant_box_indices = where(iou, overlaps);
//
//       sorted_boxes = remove(sorted_boxes, redundant_box_indices);
//     }
//
//     return filtered_boxes;
//   }
//
