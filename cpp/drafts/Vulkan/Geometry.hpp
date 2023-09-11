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

#pragma once

#include <vulkan/vulkan.h>

#include <vector>

#include <Eigen/Core>


struct Vertex
{
  Eigen::Vector2f pos;
  Eigen::Vector3f color;

  static auto get_binding_description() -> VkVertexInputBindingDescription
  {
    VkVertexInputBindingDescription binding_description{};
    binding_description.binding = 0;
    binding_description.stride = sizeof(Vertex);
    binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return binding_description;
  }

  static auto get_attribute_descriptions()
      -> std::array<VkVertexInputAttributeDescription, 2>
  {
    auto attribute_descriptions =
        std::array<VkVertexInputAttributeDescription, 2>{};

    for (auto i = 0u; i != attribute_descriptions.size(); ++i)
    {
      attribute_descriptions[0].binding = 0;
      attribute_descriptions[0].location = i;
      attribute_descriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
      attribute_descriptions[0].offset = offsetof(Vertex, pos);
    }


    return attribute_descriptions;
  }
};

const auto vertices = std::vector<Vertex>{
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},  //
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},   //
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}   //
};
