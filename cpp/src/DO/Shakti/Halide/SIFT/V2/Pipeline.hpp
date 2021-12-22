#pragma once

#include <DO/Sara/Core/Timer.hpp>

#include <DO/Shakti/Halide/SIFT/V2/ExtremumDataStructures.hpp>


namespace DO::Shakti::HalideBackend::v2 {

  struct SiftOctaveParameters
  {
    //! @brief Gaussian octave.
    //! @{
    float scale_camera = 1.f;
    float scale_initial = 1.2f;
    float scale_factor = std::pow(2.f, 1 / 1.f);
    int scale_count = 1;
    int gaussian_truncation_factor = 4;
    //! @}

    //! @brief Extremum detection thresholds.
    float edge_ratio = 10.0f;
    float extremum_thres = 0.01f;

    //! @brief Dominant gradient orientations.
    int num_orientation_bins = 36;
    float ori_gaussian_truncation_factor = 3.f;
    float scale_multiplying_factor = 1.5f;
    float peak_ratio_thres = 0.8f;

    std::vector<float> scales;
    std::vector<float> sigmas;
    std::vector<Halide::Runtime::Buffer<float>> kernels;

    inline auto set_scale_count(int n) -> void
    {
      scale_count = n;
      scale_factor = std::pow(2.f, 1.f / n);
    }

    auto initialize_gaussian_kernels() -> void;
  };


  struct SiftOctavePipeline
  {
    using GradientBuffer = std::array<Halide::Runtime::Buffer<float>, 2>;

    bool profile = false;
    Sara::Timer timer;

    std::vector<Halide::Runtime::Buffer<float>> gaussians;
    std::vector<Halide::Runtime::Buffer<float>> dogs;
    std::vector<Halide::Runtime::Buffer<std::int8_t>> extrema_maps;
    std::vector<GradientBuffer> gradients;

    std::vector<v2::QuantizedExtremumArray> extrema_quantized;
    std::vector<v2::ExtremumArray> extrema;

    std::vector<v2::DominantOrientationDenseMap>
        dominant_orientation_dense_maps;
    std::vector<v2::DominantOrientationSparseMap>
        dominant_orientation_sparse_maps;
    std::vector<v2::OrientedExtremumArray> extrema_oriented;

    // Pipeline parameters.
    SiftOctaveParameters params;

    enum class FirstAction : std::uint8_t
    {
      Convolve = 0,
      Downscale = 1,
    };

    inline SiftOctavePipeline() = default;

    inline SiftOctavePipeline(const SiftOctaveParameters& params_)
      : params{params_}
    {
    }

    inline auto tic() -> void
    {
      if (profile)
        timer.restart();
    }

    inline auto toc(const std::string& what) -> void
    {
      if (profile)
      {
        const auto elapsed = timer.elapsed_ms();
        SARA_DEBUG << "[" << what << "] " << elapsed << " ms" << std::endl;
      }
    }

    auto initialize_buffers(std::int32_t num_scales, std::int32_t w,
                            std::int32_t h) -> void;

    auto feed(Halide::Runtime::Buffer<float>& input,
              FirstAction first_action = FirstAction::Convolve) -> void;

    auto compress_quantized_extrema_maps() -> void;

    auto refine_extrema() -> void;

    auto compute_dominant_orientations() -> void;

    auto compress_dominant_orientations() -> void;

    auto populate_oriented_extrema() -> void;

    auto gaussian_view(int i) -> sara::ImageView<float>;

    auto dog_view(int i) -> sara::ImageView<float>;

    auto extrema_map_view(int i) -> sara::ImageView<std::int8_t>;
  };


  struct SiftPyramidPipeline
  {
    bool profile = false;
    Sara::Timer timer;

    float scale_initial = 1.6f;
    int image_padding_size = 1;

    // Normal.
    float scale_camera = 1;
    int start_octave_index = 0;

    // Overkill but possible.
    // float scale_camera = 0.25;
    // int start_octave_index = -2;

    // Ultra options.
    // float scale_camera = 0.5;
    // int start_octave_index = -1;

    Halide::Runtime::Buffer<float> input_rescaled;
    std::vector<SiftOctavePipeline> octaves;

    inline auto tic() -> void
    {
      if (profile)
        timer.restart();
    }

    inline auto toc(const std::string& what) -> void
    {
      if (profile)
      {
        const auto elapsed = timer.elapsed_ms();
        SARA_DEBUG << "[" << what << "] " << elapsed << " ms" << std::endl;
      }
    }

    auto initialize(int start_octave, int num_scales_per_octave, int width,
                    int height) -> void;

    auto feed(Halide::Runtime::Buffer<float>& input) -> void;

    auto octave_scaling_factor(int o) const -> float;

    auto input_rescaled_view() -> sara::ImageView<float>;
  };

}  // namespace DO::Shakti::HalideBackend::v2
