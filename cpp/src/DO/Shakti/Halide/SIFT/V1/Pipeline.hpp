#pragma once

#include <DO/Shakti/Halide/Differential.hpp>
#include <DO/Shakti/Halide/LocalExtrema.hpp>
#include <DO/Shakti/Halide/Pyramids.hpp>
#include <DO/Shakti/Halide/RefineExtrema.hpp>
#include <DO/Shakti/Halide/Utilities.hpp>

#include <DO/Shakti/Halide/DominantGradientOrientations.hpp>
#include <DO/Shakti/Halide/Resize.hpp>
#include <DO/Shakti/Halide/SIFT.hpp>


namespace DO::Shakti::HalideBackend {

  struct SIFTExtractor
  {
    struct Parameters
    {
      //! @brief Pyramid construction.
      int initial_pyramid_octave = 0;

      //! @brief Extrema detection thresholds.
      //! @{
      float edge_ratio_thres = 10.f;
      float extremum_thres = 0.01f;  // 0.03f;
      //! @}

      //! @brief Dominant gradient orientations.
      //! @{
      int num_orientation_bins = 36;
      float gaussian_truncation_factor = 3.f;
      float scale_multiplying_factor = 1.5f;
      float peak_ratio_thres = 0.0f;
      //! @}

      //! @brief SIFT descriptor parameters.
      //! @{
      float bin_length_in_scale_unit = 3.f;
      int N = 4;
      int O = 8;
      //! @}
    };

    struct Pipeline
    {
      Sara::ImagePyramid<float> gaussian_pyramid;
      Sara::ImagePyramid<float> dog_pyramid;
      Sara::ImagePyramid<std::int8_t> dog_extrema_pyramid;
      std::array<Sara::ImagePyramid<float>, 2> gradient_pyramid;

      Pyramid<QuantizedExtremumArray> extrema_quantized;
      Pyramid<ExtremumArray> extrema;

      Pyramid<DominantOrientationDenseMap> dominant_orientations_dense;
      Pyramid<DominantOrientationMap> dominant_orientations;

      // The DoG extremum keypoints.
      Pyramid<OrientedExtremumArray> oriented_extrema;
      // The SIFT descriptors.
      Pyramid<Sara::Tensor_<float, 4>> descriptors;
      Pyramid<Sara::Tensor_<float, 2>> descriptors_v2;
      Pyramid<Sara::Tensor_<float, 3>> descriptors_v3;

      auto num_keypoints() const
      {
        return std::accumulate(
            oriented_extrema.dict.begin(), oriented_extrema.dict.end(), 0ul,
            [](auto val, const auto& kv) { return val + kv.second.size(); });
      }
    };

    Sara::Timer timer;
    Parameters params;
    Pipeline pipeline;

    auto operator()(Sara::ImageView<float>& image)
    {
      timer.restart();
      const auto pyr_params =
          Sara::ImagePyramidParams{params.initial_pyramid_octave};
      pipeline.gaussian_pyramid = gaussian_pyramid(image, pyr_params);
      SARA_DEBUG << "Gaussian pyramid = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      timer.restart();
      pipeline.dog_pyramid = subtract_pyramid(pipeline.gaussian_pyramid);
      SARA_DEBUG << "DoG pyramid = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      timer.restart();
      pipeline.dog_extrema_pyramid = local_scale_space_extrema(  //
          pipeline.dog_pyramid,                                  //
          params.edge_ratio_thres,                               //
          params.extremum_thres);
      SARA_DEBUG << "DoG extrema = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      timer.restart();
      std::tie(pipeline.gradient_pyramid[0],                     //
               pipeline.gradient_pyramid[1]) =                   //
          halide::polar_gradient_2d(pipeline.gaussian_pyramid);  //
      SARA_DEBUG << "Gradient pyramid = " << timer.elapsed_ms() << " ms"
                 << std::endl;

      // Populate the DoG extrema.
      pipeline.extrema_quantized = populate_local_scale_space_extrema(  //
          pipeline.dog_extrema_pyramid);

      // Refine the scale-space localization of each extremum.
      pipeline.extrema = refine_scale_space_extrema(  //
          pipeline.dog_pyramid,                       //
          pipeline.extrema_quantized);                //

      // Estimate the dominant gradient orientations.
      timer.restart();
      dominant_gradient_orientations(pipeline.gradient_pyramid[0],          //
                                     pipeline.gradient_pyramid[1],          //
                                     pipeline.extrema,                      //
                                     pipeline.dominant_orientations_dense,  //
                                     params.num_orientation_bins,           //
                                     params.gaussian_truncation_factor,     //
                                     params.scale_multiplying_factor,       //
                                     params.peak_ratio_thres);
      SARA_DEBUG << "Dominant gradient orientations = " << timer.elapsed_ms()
                 << " ms" << std::endl;

      timer.restart();
      pipeline.dominant_orientations =
          compress(pipeline.dominant_orientations_dense);

      pipeline.oriented_extrema = to_oriented_extremum_array(
          pipeline.extrema, pipeline.dominant_orientations);
      SARA_DEBUG << "Populating oriented extrema = " << timer.elapsed_ms()
                 << " ms" << std::endl;

// #define SIFT_V1
// #define SIFT_V2
// #define SIFT_V3
#define SIFT_V4
#if defined(SIFT_V1)
      SARA_DEBUG << "RUNNING SIFT V1..." << std::endl;
      timer.restart();
      pipeline.descriptors = v1::compute_sift_descriptors(
          pipeline.gradient_pyramid[0], pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema, params.bin_length_in_scale_unit, params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#elif defined(SIFT_V2)
      SARA_DEBUG << "RUNNING SIFT V2..." << std::endl;
      timer.restart();
      pipeline.descriptors_v2 = v2::compute_sift_descriptors(
          pipeline.gradient_pyramid[0], pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema, params.bin_length_in_scale_unit, params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#elif defined(SIFT_V3)
      SARA_DEBUG << "RUNNING SIFT V3..." << std::endl;
      timer.restart();
      pipeline.descriptors_v3 = v3::compute_sift_descriptors(
          pipeline.gradient_pyramid[0], pipeline.gradient_pyramid[1],
          pipeline.oriented_extrema, params.bin_length_in_scale_unit, params.N,
          params.O);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#elif defined(SIFT_V4)
      SARA_DEBUG << "RUNNING SIFT V4..." << std::endl;
      timer.restart();
      pipeline.descriptors_v3 = v4::compute_sift_descriptors(  //
          pipeline.gradient_pyramid[0],                        //
          pipeline.gradient_pyramid[1],                        //
          pipeline.oriented_extrema);
      SARA_DEBUG << "SIFT descriptors = " << timer.elapsed_ms() << " ms"
                 << std::endl;
#endif
    }
  };

}  // namespace DO::Shakti::HalideBackend

