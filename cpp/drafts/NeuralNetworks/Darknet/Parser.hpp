// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>
#include <drafts/NeuralNetworks/Darknet/Network.hpp>

#include <boost/filesystem.hpp>


namespace DO::Sara::Darknet {

  struct NetworkParser
  {
    auto read_line(std::ifstream& file, std::string& line) const -> bool;

    auto is_section(const std::string& line) const -> bool;

    auto is_comment(const std::string& line) const -> bool;

    auto section_name(const std::string& line) const -> std::string;

    auto make_new_layer(const std::string& layer_type,
                        std::vector<std::unique_ptr<Layer>>& nodes) const
        -> void;

    auto finish_layer_init(std::vector<std::unique_ptr<Layer>>& nodes) const
        -> void;

    auto parse_config_file(const std::string& cfg_filepath) const
        -> std::vector<std::unique_ptr<Layer>>;
  };

  //! Parses the .weights file from Darknet library.
  //!
  //! IMPORTANT: this is not portable and is guaranteed to work only on
  //! little-endian machine and 64-bit architecture.
  struct NetworkWeightLoader
  {
    FILE* fp = nullptr;

    int major;
    int minor;
    int revision;
    uint64_t seen;
    int transpose;

    bool debug = false;

    NetworkWeightLoader() = default;

    NetworkWeightLoader(const std::string& filepath);

    ~NetworkWeightLoader();

    auto load(std::vector<std::unique_ptr<Layer>>& net) -> void;
  };


  auto load_yolov4_tiny_model(const boost::filesystem::path& model_dir_path,
                              const int version) -> Network;

  auto load_yolo_model(const boost::filesystem::path& model_dir_path,
                       const int version) -> Network;

}  // namespace DO::Sara::Darknet
