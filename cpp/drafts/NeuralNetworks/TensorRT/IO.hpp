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

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>

#include <fstream>
#include <sstream>


namespace DO::Sara::TensorRT {

  //! @brief Helper function for serializing TensorRT plugins.
  //!
  //! N.B.: this implementation taken from GitHub implies that TensorRT will
  //! not guarantee portability w.r.t. endianness.
  template <typename T>
  inline auto write_to_buffer(char*& buffer, const T& val) -> void
  {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  //! @brief Helper function for deserializing TensorRT plugins.
  //!
  //! N.B.: this implementation taken from GitHub implies that TensorRT will
  //! not guarantee portability w.r.t. endianness.
  template <typename T>
  inline auto read_from_buffer(const char*& buffer) -> T
  {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
  }


  auto serialize_network_into_plan(const BuilderUniquePtr& network_builder,
                                   const NetworkUniquePtr& network,
                                   const bool use_fp16 = false)
      -> HostMemoryUniquePtr;

  auto write_plan(const HostMemoryUniquePtr& model_weights,
                  const std::string& model_weights_filepath) -> void;

}  // namespace DO::Sara::TensorRT
