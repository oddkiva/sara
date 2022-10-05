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

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <fstream>
#include <sstream>


namespace DO::Sara::TensorRT {

  auto serialize_network_into_plan(const BuilderUniquePtr& network_builder,
                                   const NetworkUniquePtr& network,
                                   const bool use_fp16) -> HostMemoryUniquePtr
  {
    // Create an inference configuration object.
    auto config = ConfigUniquePtr{network_builder->createBuilderConfig(),  //
                                  &config_deleter};
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 32u);
    // config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    if (use_fp16)
      config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Serialize the network definition and weights for TensorRT.
    auto plan = HostMemoryUniquePtr{
        network_builder->buildSerializedNetwork(*network, *config),  //
        host_memory_deleter};
    if (plan.get() == nullptr)
      throw std::runtime_error{"Failed to build TensorRT plan!"};

    return plan;
  }

  auto write_plan(const HostMemoryUniquePtr& model_weights,
                  const std::string& model_weights_filepath) -> void
  {
    // Save in the disk.
    auto model_weights_stream = std::stringstream{};
    model_weights_stream.seekg(0, model_weights_stream.beg);
    model_weights_stream.write(
        reinterpret_cast<const char*>(model_weights->data()),
        model_weights->size());

    auto model_weights_file = std::ofstream{
        model_weights_filepath, std::ofstream::out | std::ofstream::binary};
    if (!model_weights_file)
      throw std::runtime_error{"Failed to create model weights file!"};
    model_weights_file << model_weights_stream.rdbuf();
  }

}  // namespace DO::Sara::TensorRT
