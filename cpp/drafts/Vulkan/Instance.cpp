// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2023 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <drafts/Vulkan/Instance.hpp>


namespace svk = DO::Shakti::Vulkan;

auto svk::list_available_instance_layers() -> std::vector<VkLayerProperties>
{
  auto layer_count = std::uint32_t{};
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  if (layer_count == 0)
    return {};

  auto available_layers = std::vector<VkLayerProperties>(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  return available_layers;
}
