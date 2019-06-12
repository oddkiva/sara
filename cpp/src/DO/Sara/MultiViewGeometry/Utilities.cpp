#include <DO/Sara/MultiViewGeometry/Utilities.hpp>


namespace DO { namespace Sara {

  auto range(int n) -> Tensor_<int, 1>
  {
    auto indices = Tensor_<int, 1>{n};
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  auto random_samples(int num_samples,      //
                      int sample_size,      //
                      int num_data_points)  //
      -> Tensor_<int, 2>
  {
    auto indices = range(num_data_points);

    auto samples = Tensor_<int, 2>{{sample_size, num_samples}};
    for (int i = 0; i < sample_size; ++i)
      samples[i].flat_array() =
          shuffle(indices).flat_array().head(num_samples);

    samples = samples.transpose({1, 0});

    return samples;
  }

  // Data transformations.
  auto extract_centers(const std::vector<OERegion>& features)
      -> Tensor_<float, 2>
  {
    auto centers = Tensor_<float, 2>{{int(features.size()), 2}};
    auto mat = centers.matrix();

    for (auto i = 0; i < centers.size(0); ++i)
      mat.row(i) = features[i].center().transpose();

    return centers;
  }

  auto to_point_indices(const TensorView_<int, 2>& samples,
                        const TensorView_<int, 2>& matches)  //
      -> Tensor_<int, 3>
  {
    const auto num_samples = samples.size(0);
    const auto sample_size = samples.size(1);

    auto point_indices = Tensor_<int, 3>{{num_samples, sample_size, 2}};
    for (auto s = 0; s < num_samples; ++s)
      for (auto m = 0; m < sample_size; ++m)
        point_indices[s][m].flat_array() = matches[samples(s, m)].flat_array();

    return point_indices;
  }

} /* namespace Sara */
} /* namespace DO */
