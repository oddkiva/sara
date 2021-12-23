// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2020-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

/*!
 *  Nothing changed from V4 except that I don't want to spend time debugging
 *  with buffer reshaping with Halide API with the GPU device memory.
 *
 *  Every image buffers is seen as 4D tensors to avoid reimplementing the same
 *  algorithms over and over.
 */

#include <array>

#include <DO/Shakti/Halide/Components/SIFT.hpp>


namespace {

  using namespace Halide;

  class SIFTv5 : public Generator<SIFTv5>
  {
  public:
    GeneratorParam<int> tile_k{"tile_k", 8};
    GeneratorParam<int> tile_u{"tile_u", 4};
    GeneratorParam<int> tile_v{"tile_v", 4};
    GeneratorParam<int> tile_ji{"tile_ji", 16};
    GeneratorParam<int> tile_o{"tile_o", 8};

    //! @brief Variables.
    //!
    //! @brief Keypoint index.
    Var k{"k"}, ko{"ko"}, ki{"ki"};
    //! @brief Normalized patch coordinates.
    Var u{"u"}, uo{"uo"}, ui{"ui"};
    Var v{"v"}, vo{"vo"}, vi{"vi"};
    //! @brief SIFT bin coordinates.
    Var o{"o"}, oo{"oo"}, oi{"oi"};
    Var ji{"ji"}, jio{"jio"}, jii{"jii"};

    //! @brief Intermediate compute functions.
    DO::Shakti::HalideBackend::SIFT sift;
    //! @brief Precomputed weight functions.
    Halide::Func gradient_weight_fn{"gradient_weights"};
    Halide::Func spatial_weight_fn{"spatial_weights"};
    Halide::Func normalized_gradient_fn{"normalized_gradients"};
    Halide::Func descriptors_unnormalized{"descriptor_unnormalized"};

    //! @brief Input data.
    Input<Buffer<float>[2]> polar_gradient { "gradients", 4 };
    Input<Buffer<float>[4]> xyst { "xyst", 1 };

    //! @brief Output data.
    Output<Buffer<float>> descriptors{"SIFT", 3};

    void generate()
    {
      const auto& mag = polar_gradient[0];
      const auto& ori = polar_gradient[1];
      const auto& mag_fn_ext = BoundaryConditions::constant_exterior(mag, 0);
      const auto& ori_fn_ext = BoundaryConditions::constant_exterior(ori, 0);

      const auto x = xyst[0](k);
      const auto y = xyst[1](k);
      const auto s = xyst[2](k);
      const auto theta = xyst[3](k);

      namespace halide = DO::Shakti::HalideBackend;

      // Already we can precompute the following functions.
      //
      // This function is used to reweigh the image gradient magnitudes.
      gradient_weight_fn(u, v) = sift.gradient_weight(u, v);
      // This function is used in the trilinear interpolation for the
      // accumulation of histogram of gradients.
      spatial_weight_fn(u, v) = sift.spatial_weight(u, v);

      // TODO
      //
      // The original SIFT implementation by David Lowe scans in the original
      // image domain. But this is not easy to implement efficiently on the GPU
      // as-is.
      //
      // Instead, we calculate the normalized patch of gradients, after which we
      // calculate the NxN histogram of gradients.
      //
      // So we have to check whether:
      // - the different algorithm gives close results to the original SIFT
      //   algorithm implemented by David Lowe.
      // - the different algorithm gives good results on the matching.
      //
      // Precompute the normalized patch of gradients.
      normalized_gradient_fn(u, v, k) = sift.normalized_gradient_sample_v2(  //
          u, v,                                                              //
          mag_fn_ext,                                                        //
          ori_fn_ext,                                                        //
          gradient_weight_fn,                                                //
          x, y, s, theta                                                     //
      );

#define NORMALIZE_SIFT
#ifdef NORMALIZE_SIFT
      descriptors_unnormalized(o, ji, k) = 0.f;
      sift.accumulate_subhistogram_v3(descriptors_unnormalized,  //
                                      ji, k,                     //
                                      normalized_gradient_fn,
                                      spatial_weight_fn);

      sift.normalize_histogram(descriptors_unnormalized, o, ji, k);
      descriptors(o, ji, k) = sift.hist_illumination_invariant(o, ji, k);
#else
      descriptors(o, ji, k) = 0.f;
      sift.accumulate_subhistogram_v3(descriptors,  //
                                      ji, k,        //
                                      normalized_gradient_fn,
                                      spatial_weight_fn);
#endif
    }

    void schedule()
    {
      // GPU schedule.
      if (!get_target().has_gpu_feature())
        throw std::runtime_error{"GPU implementation only!"};

      gradient_weight_fn.compute_root();
      spatial_weight_fn.compute_root();

      normalized_gradient_fn.compute_root();
      normalized_gradient_fn.gpu_tile(  //
          u, v, k,                      //
          uo, vo, ko,                   //
          ui, vi, ki,                   //
          tile_u, tile_v, tile_k,       //
          Halide::TailStrategy::GuardWithIf);

#ifdef NORMALIZE_SIFT
      descriptors_unnormalized.compute_root();
      descriptors_unnormalized.gpu_tile(o, ji, k,                 //
                                        oo, jio, ko,              //
                                        oi, jii, ki,              //
                                        tile_o, tile_ji, tile_k,  //
                                        TailStrategy::GuardWithIf);
#endif

      descriptors.gpu_tile(o, ji, k,                 //
                           oo, jio, ko,              //
                           oi, jii, ki,              //
                           tile_o, tile_ji, tile_k,  //
                           TailStrategy::GuardWithIf);
    }
  };

}  // namespace


HALIDE_REGISTER_GENERATOR(SIFTv5, shakti_sift_descriptor_gpu_v5)
