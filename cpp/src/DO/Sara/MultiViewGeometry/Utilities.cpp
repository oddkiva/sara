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

} /* namespace Sara */
} /* namespace DO */
