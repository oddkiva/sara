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

#define BOOST_TEST_MODULE "Halide Backend/SIFT components"

#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Shakti/Halide/RuntimeUtilities.hpp>

#include <boost/test/unit_test.hpp>

#include "shakti_brute_force_nn_l2_32f_cpu.h"
#include "shakti_brute_force_nn_l2_32f_gpu.h"


namespace sara = DO::Sara;
namespace hl = DO::Shakti::Halide;

BOOST_AUTO_TEST_CASE(test_brute_force_nn)
{
  auto d1 = sara::Tensor_<float, 2>{16000, 128};
  for (auto i = 0; i < d1.rows(); ++i)
    d1[i].flat_array().fill(rand() % 1000);
  d1.flat_array() /= 1000;

  auto d2 = d1;
  d2.flat_array() += 0.1f;

  auto dist = Eigen::VectorXf(d1.size(0));
  auto nn = Eigen::VectorXi(d1.size(0));

  auto d1_ = hl::as_runtime_buffer(d1);
  auto d2_ = hl::as_runtime_buffer(d2);
  auto dist_ = hl::as_runtime_buffer(dist);
  auto nn_ = hl::as_runtime_buffer(nn);

  SARA_CHECK(d1_.dim(0).extent());
  SARA_CHECK(d1_.dim(1).extent());

  sara::tic();
  d1_.set_host_dirty();
  d2_.set_host_dirty();
  dist_.set_host_dirty();
  nn_.set_host_dirty();
  shakti_brute_force_nn_l2_32f_gpu(d1_, d2_, dist_, nn_);
  dist_.copy_to_host();
  nn_.copy_to_host();
  sara::toc("Brute-Force NN");

  // SARA_DEBUG << std::endl << dist << std::endl;
  // SARA_DEBUG << std::endl << nn << std::endl;
}
