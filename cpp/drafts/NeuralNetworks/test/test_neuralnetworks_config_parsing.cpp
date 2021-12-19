// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "NeuralNetworks/Yolo Configuration Parsing"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>
#include <drafts/NeuralNetworks/Darknet/Parser.hpp>

#include <boost/endian.hpp>
#include <boost/test/unit_test.hpp>


namespace sara = DO::Sara;


BOOST_AUTO_TEST_SUITE(TestLayers)

BOOST_AUTO_TEST_CASE(test_yolov4_tiny_config_parsing)
{
  namespace fs = boost::filesystem;

  const auto data_dir_path =
      fs::canonical(fs::path{src_path("../../../../data")});
  const auto cfg_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.cfg";
  const auto weights_filepath =
      data_dir_path / "trained_models" / "yolov4-tiny.weights";
  BOOST_CHECK(fs::exists(cfg_filepath));

  auto net = sara::Darknet::NetworkParser{}.parse_config_file(cfg_filepath.string());
  if (fs::exists(weights_filepath))
    sara::Darknet::NetworkWeightLoader{weights_filepath.string()}.load(net);
}

BOOST_AUTO_TEST_SUITE_END()
