#pragma once

#include <drafts/NeuralNetworks/Darknet/YoloUtilities.hpp>


namespace DO::Sara::Darknet {

  auto get_yolo_boxes(const TensorView_<float, 3>& output,
                      const std::vector<int>& box_sizes_prior,
                      const std::vector<int>& masks,
                      const Eigen::Vector2i& network_input_sizes,
                      const Eigen::Vector2i& original_sizes,
                      float objectness_threshold) -> std::vector<YoloBox>
  {
    const Eigen::Vector2f scale = original_sizes.cast<float>().array() /
                                  network_input_sizes.cast<float>().array();

    auto boxes = std::vector<YoloBox>{};
    const auto num_boxes = static_cast<int>(masks.size());
    for (auto box = 0; box < num_boxes; ++box)
    {
      // Box center
      const auto rel_x = output[box * 85 + 0];
      const auto rel_y = output[box * 85 + 1];
      // Box log sizes
      const auto log_w = output[box * 85 + 2];
      const auto log_h = output[box * 85 + 3];
      // Objectness probability.
      const auto objectness = output[box * 85 + 4];

      // Fetch the box size prior.
      const auto& w_prior = box_sizes_prior[2 * masks[box] + 0];
      const auto& h_prior = box_sizes_prior[2 * masks[box] + 1];

      for (auto i = 0; i < output.size(1); ++i)
        for (auto j = 0; j < output.size(2); ++j)
        {
          if (objectness(i, j) < objectness_threshold)
            continue;

          // This is the center of the box and not the top-left corner of the
          // box.
          auto xy = Eigen::Vector2f{
              (j + rel_x(i, j)) / output.size(2),  //
              (i + rel_y(i, j)) / output.size(1)   //
          };
          xy.array() *= original_sizes.cast<float>().array();

          // Exponentiate and rescale to get the box sizes.
          auto wh = Eigen::Vector2f{log_w(i, j), log_h(i, j)};
          wh.array() = wh.array().exp();
          wh(0) *= w_prior * scale.x();
          wh(1) *= h_prior * scale.y();

          // The final box.
          const auto xywh =
              (Eigen::Vector4f{} << (xy - wh * 0.5f), wh).finished();

          // The probabilities.
          const auto obj_prob = objectness(i, j);
          auto class_probs = Eigen::VectorXf{80};
          for (auto c = 0; c < 80; ++c)
            class_probs(c) = output[box * 85 + 5 + c](i, j);

          boxes.push_back({xywh, obj_prob, class_probs});
        }
    }

    return boxes;
  }

  // Simple greedy NMS based on area IoU criterion.
  auto nms(const std::vector<YoloBox>& detections, float iou_threshold)
      -> std::vector<YoloBox>
  {
    auto detections_sorted = detections;
    std::sort(detections_sorted.begin(), detections_sorted.end(),
              [](const auto& a, const auto& b) {
                return a.objectness_prob > b.objectness_prob;
              });

    auto detections_filtered = std::vector<YoloBox>{};
    detections_filtered.reserve(detections.size());

    for (const auto& d : detections_sorted)
    {
      if (detections_filtered.empty())
      {
        detections_filtered.push_back(d);
        continue;
      }

      auto boxes_kept = Eigen::MatrixXf(detections_filtered.size(), 4);
      for (auto i = 0u; i < detections_filtered.size(); ++i)
        boxes_kept.row(i) = detections_filtered[i].box.transpose();

      const auto x1 = boxes_kept.col(0);
      const auto y1 = boxes_kept.col(1);
      const auto w = boxes_kept.col(2);
      const auto h = boxes_kept.col(3);

      const auto x2 = x1 + w;
      const auto y2 = y1 + h;

      // Intersection.
      const auto inter_x1 = x1.array().max(d.box(0));
      const auto inter_y1 = y1.array().max(d.box(1));
      const auto inter_x2 = x2.array().min(d.box(0) + d.box(2));
      const auto inter_y2 = y2.array().min(d.box(1) + d.box(3));
      const auto intersect = (inter_x1 <= inter_x2) && (inter_y1 <= inter_y2);

      // Intersection areas
      const Eigen::ArrayXf inter_area =
          intersect.cast<float>() *
          ((inter_x2 - inter_x1) * (inter_y2 - inter_y1));

      // Union areas.
      const Eigen::ArrayXf union_area =
          w.array() * h.array() + d.box(2) * d.box(3) - inter_area;

      // IoU
      const Eigen::ArrayXf iou = inter_area / union_area;

      const auto valid = (iou < iou_threshold).all();
      if (valid)
        detections_filtered.push_back(d);
    }

    return detections_filtered;
  }

}  // namespace DO::Sara::Darknet
