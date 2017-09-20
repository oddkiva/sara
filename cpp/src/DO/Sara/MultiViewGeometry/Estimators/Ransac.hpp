#pragma once

#include <random>


namespace DO { namespace Sara {

  /*
   * generate list of random samples | normalize | seven_point_algorithm | filter_best_distance
   */

  template <typename Sample>
  struct RandomSampleGeneratorIterator
  {
    RandomSampleGeneratorIterator(size_t buffer_size = 1)
      : buffer{buffer_size}
    {
    }

    auto operator++()() const;

    std::vector<Sample> buffer;
  };


  //! @brief Random Sample Consensus algorithm from Fischler and Bolles 1981.
  template <typename Estimator, typename Distance>
  struct Ransac
  {
    using scalar_type = typename Estimator::scalar_type;
    using out_parameter_type = typename Estimator::out_parameter_type;

    Ransac(scalar_type outlier_threshold, scalar_type max_outlier_ratio = 0.5,
           scalar_type min_probability = 0.99);

    template <typename SampleIterator>
    auto operator()(SampleIterator begin, SampleIterator end) -> const out_parameter_type&;

    //! @{
    //! @brief Computational functor.
    Estimator estimator;
    Distance distance;
    //! @}

    out_parameter_type estimated_parameter;

    scalar_type outlier_threshold;
    scalar_type max_outlier_ratio;
    scalar_type min_probability;
  };

} /* namespace Sara */
} /* namespace DO */
