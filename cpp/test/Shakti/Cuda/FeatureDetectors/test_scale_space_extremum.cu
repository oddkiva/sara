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

#define BOOST_TEST_MODULE "Shakti/CUDA/FeatureDetectors/Octave"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Core/TensorDebug.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/Octave.hpp>
#include <DO/Shakti/Cuda/FeatureDetectors/ScaleSpaceExtremum.hpp>
#include <DO/Shakti/Cuda/Utilities.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = DO::Shakti::Cuda;


BOOST_AUTO_TEST_CASE(test_extremum_localization)
{
  static constexpr auto w = 3;
  static constexpr auto h = 3;
  static constexpr auto d = 3;

  auto h_octave = sara::Image<float, 3>{w, h, d};
  auto h_data = sara::tensor_view(h_octave);
  h_octave.flat_array().fill(0);
  h_data[0].matrix() << 1, 2, 4,
                        1, 5, 2,
                        2, 3, 1,
  h_data[1].matrix() << 2,  4, 5,
                        1, 10, 2,
                        2,  3, 1,
  h_data[2].matrix().fill(0);

  SARA_CHECK(h_octave.sizes().transpose());

  SARA_DEBUG << "h_data[0] = \n" << h_data[0].matrix() << std::endl;
  SARA_DEBUG << "h_data[1] = \n" << h_data[1].matrix() << std::endl;
  SARA_DEBUG << "h_data[2] = \n" << h_data[2].matrix() << std::endl;

  // Initialize the octave CUDA surface.
  auto d_octave = sc::Octave<float>(w, h, d);
  d_octave.array().copy_from(h_octave);
  d_octave.init_surface();

#define CHECK_DEVICE_OCTAVE_CONTENTS
#ifdef CHECK_DEVICE_OCTAVE_CONTENTS
  auto h_octave_copy = sara::Image<float, 3>{w, h, d};
  d_octave.array().copy_to(h_octave_copy);

  auto h_data_copy = sara::tensor_view(h_octave_copy);
  SARA_DEBUG << "CHECK DEVICE OCTAVE" << std::endl;
  SARA_DEBUG << "h_data[0] = \n" << h_data_copy[0].matrix() << std::endl;
  SARA_DEBUG << "h_data[1] = \n" << h_data_copy[1].matrix() << std::endl;
  SARA_DEBUG << "h_data[2] = \n" << h_data_copy[2].matrix() << std::endl;
#endif

  auto d_extremum_map = shakti::MultiArray<std::int8_t, 1>{w * h * d};
  auto h_extremum_map = sara::Image<std::int8_t, 3>{w, h, d};

  sc::compute_scale_space_extremum_map(d_octave, d_extremum_map, 0.f, 0.f);
  d_extremum_map.copy_to_host(h_extremum_map.data());

  auto h_ext_tensor = sara::tensor_view(h_extremum_map);
  SARA_DEBUG << "CHECK DEVICE EXTREMUM MAP" << std::endl;
  SARA_DEBUG << "h_ext_tensor[0] = \n" << h_ext_tensor[0].matrix().cast<int>() << std::endl;
  SARA_DEBUG << "h_ext_tensor[1] = \n" << h_ext_tensor[1].matrix().cast<int>() << std::endl;
  SARA_DEBUG << "h_ext_tensor[2] = \n" << h_ext_tensor[2].matrix().cast<int>() << std::endl;

  auto gt_extremum_map = sara::Image<std::int8_t, 3>{w, h, d};
  gt_extremum_map.flat_array().fill(0);
  gt_extremum_map(1, 1, 1) = 1;

  BOOST_CHECK(
      (gt_extremum_map.flat_array() == h_extremum_map.flat_array()).all());

  auto d_extrema = sc::compress_extremum_map(d_extremum_map);
  BOOST_CHECK_EQUAL(d_extrema.indices.size(), 1u);
  BOOST_CHECK_EQUAL(d_extrema.types.size(), 1u);
  BOOST_CHECK_EQUAL(d_extrema.indices[0], 9 + 5 - 1);
  BOOST_CHECK_EQUAL(d_extrema.types[0], 1);

  sc::initialize_extrema(d_extrema, d_octave.width(), d_octave.height(), d_octave.scale_count());
  BOOST_CHECK_EQUAL(d_extrema.x.size(), 1u);
  BOOST_CHECK_EQUAL(d_extrema.y.size(), 1u);
  BOOST_CHECK_EQUAL(d_extrema.s.size(), 1u);
  BOOST_CHECK_EQUAL(d_extrema.x[0], 1);
  BOOST_CHECK_EQUAL(d_extrema.y[0], 1);
  BOOST_CHECK_EQUAL(d_extrema.s[0], 1);

  sc::refine_extrema(d_octave, d_extrema);
  const auto h_extrema = d_extrema.copy_to_host();
  BOOST_CHECK_EQUAL(h_extrema.indices.size(), 1u);
  BOOST_CHECK_EQUAL(h_extrema.x.size(), 1u);
  BOOST_CHECK_EQUAL(h_extrema.y.size(), 1u);
  BOOST_CHECK_EQUAL(h_extrema.s.size(), 1u);
  BOOST_CHECK_EQUAL(h_extrema.types.size(), 1u);
  BOOST_CHECK_EQUAL(h_extrema.types.size(), 1u);

  BOOST_CHECK_EQUAL(h_extrema.indices[0], 9 + 5 - 1);
  BOOST_CHECK_EQUAL(h_extrema.types[0], 1);
  BOOST_CHECK_LE(std::abs(h_extrema.x[0] - 1), .2f);
  BOOST_CHECK_LE(std::abs(h_extrema.y[0] - 1), .2f);
  BOOST_CHECK_LE(std::abs(h_extrema.s[0] - 1), .2f);
  BOOST_CHECK_GE(h_extrema.values[0], 10);
  BOOST_CHECK_EQUAL(h_extrema.refined[0], 1);
  SARA_CHECK(h_extrema.values[0]);
  SARA_CHECK(int(h_extrema.refined[0]));
}
