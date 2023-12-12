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

#include <DO/Shakti/Vulkan/DescriptorSet.hpp>
#include <DO/Shakti/Vulkan/Device.hpp>
#include <DO/Shakti/Vulkan/RenderPass.hpp>
#include <DO/Shakti/Vulkan/Shader.hpp>

#include <array>
#include <filesystem>


namespace DO::Kalpana::Vulkan {

  //! The graphics pipeline, which is called the render pipeline in Metal.
  //!
  //! This object specifies:
  //! - what vertex shader
  //! - what fragment shader
  //! we want to use.
  struct GraphicsPipeline
  {
    class Builder;

    GraphicsPipeline() = default;

    GraphicsPipeline(const GraphicsPipeline&) = delete;

    GraphicsPipeline(GraphicsPipeline&& other)
    {
      swap(other);
    }

    auto operator=(const GraphicsPipeline&) -> GraphicsPipeline& = delete;

    auto operator=(GraphicsPipeline&& other) -> GraphicsPipeline&
    {
      swap(other);
      return *this;
    }

    ~GraphicsPipeline()
    {
      if (device == nullptr)
        return;

      if (pipeline_layout != nullptr)
      {
        SARA_DEBUG << "Destroying graphics pipeline layout...\n";
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      }

      if (pipeline != nullptr)
      {
        SARA_DEBUG << "Destroying graphics pipeline...\n";
        vkDestroyPipeline(device, pipeline, nullptr);
      }
    }

    operator VkPipeline() const
    {
      return pipeline;
    }

    auto swap(GraphicsPipeline& other) -> void
    {
      std::swap(device, other.device);
      desc_set_layout.swap(other.desc_set_layout);
      std::swap(pipeline_layout, other.pipeline_layout);
      std::swap(pipeline, other.pipeline);
    }

    VkDevice device = nullptr;
    // Model View Projection matrix stack etc.
    Shakti::Vulkan::DescriptorSetLayout desc_set_layout;
    VkPipelineLayout pipeline_layout = nullptr;
    VkPipeline pipeline = nullptr;
  };

  class GraphicsPipeline::Builder
  {
  public:
    Builder(const Shakti::Vulkan::Device& device,
            const Kalpana::Vulkan::RenderPass& render_pass)
      : device{device}
      , render_pass{render_pass}
    {
    }

    virtual ~Builder() = default;

    auto vertex_shader_path(const std::filesystem::path& source_filepath)
        -> Builder&
    {
      vertex_shader_filepath = source_filepath;
      return *this;
    }

    auto fragment_shader_path(const std::filesystem::path& source_filepath)
        -> Builder&
    {
      fragment_shader_filepath = source_filepath;
      return *this;
    }

    // 5. Data format of the vertex buffer.
    template <typename VertexDescription>
    auto vbo_data_format() -> Builder&
    {
      binding_description = VertexDescription::get_binding_description();
      attribute_descriptions = VertexDescription::get_attribute_descriptions();
      vertex_input_info = VkPipelineVertexInputStateCreateInfo{};
      vertex_input_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
      vertex_input_info.vertexBindingDescriptionCount = 1;
      vertex_input_info.pVertexBindingDescriptions = &binding_description;
      vertex_input_info.vertexAttributeDescriptionCount =
          static_cast<std::uint32_t>(attribute_descriptions.size());
      vertex_input_info.pVertexAttributeDescriptions =
          attribute_descriptions.data();

      return *this;
    }

    auto vbo_data_built_in_vertex_shader() -> Builder&
    {
      vertex_input_info = VkPipelineVertexInputStateCreateInfo{};
      vertex_input_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

      return *this;
    }

    // 6. Data type of the 3D geometry.
    //    Here a list of triangles.
    auto input_assembly_topology(
        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        -> Builder&
    {
      input_assembly = VkPipelineInputAssemblyStateCreateInfo{};
      input_assembly.sType =
          VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
      input_assembly.topology = topology;
      input_assembly.primitiveRestartEnable = VK_FALSE;

      return *this;
    }

    auto dynamic_states(const std::vector<VkDynamicState>& states) -> Builder&
    {
      _dynamic_states = states;
      dynamic_state_info = {};
      dynamic_state_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
      dynamic_state_info.dynamicStateCount =
          static_cast<std::uint32_t>(_dynamic_states.size());
      dynamic_state_info.pDynamicStates = _dynamic_states.data();

      return *this;
    }

    // Viewport: which portion of the window?
    //
    // Here we want to render on the whole window.
    auto viewport_sizes(const float w, const float h) -> Builder&
    {
      viewport = VkViewport{
          .x = 0.f,
          .y = 0.f,
          .width = w,
          .height = h,
          .minDepth = 0.f,
          .maxDepth = 1.f  //
      };

      return *this;
    }

    auto scissor_sizes(const int w, const int h) -> Builder&
    {
      scissor = VkRect2D{
          .offset = {0, 0},  //
          .extent = {static_cast<std::uint32_t>(w),
                     static_cast<std::uint32_t>(h)}  //
      };

      return *this;
    }

    auto vertex_shader_stage_info() -> VkPipelineShaderStageCreateInfo&
    {
      return shader_stage_infos[0];
    }

    auto fragment_shader_stage_info() -> VkPipelineShaderStageCreateInfo&
    {
      return shader_stage_infos[1];
    }

    virtual auto create() -> GraphicsPipeline;

  protected:
    //! @brief Boilerplate methods to call before creating the graphics
    //! pipeline.
    //! @{
    auto load_shaders() -> void;

    auto initialize_fixed_functions() -> void;
    //! @}

    //! @brief Boilerplate methods to finish creating the graphics pipeline.
    //! @{
    auto create_graphics_pipeline_layout(GraphicsPipeline& graphics_pipeline)
        -> void;

    auto create_graphics_pipeline(GraphicsPipeline& graphics_pipeline) -> void;
    //! @}

  public:
    const Shakti::Vulkan::Device& device;
    const Kalpana::Vulkan::RenderPass& render_pass;

    //! @brief Paths to shader source.
    std::filesystem::path vertex_shader_filepath;
    std::filesystem::path fragment_shader_filepath;
    std::vector<char> vertex_shader;
    std::vector<char> fragment_shader;

    //! @brief Compiled shaders.
    Shakti::Vulkan::ShaderModule vertex_shader_module;
    Shakti::Vulkan::ShaderModule fragment_shader_module;

    //! @brief The shader create infos that bind the shader modules.
    std::array<VkPipelineShaderStageCreateInfo, 2> shader_stage_infos;

    //! @brief Data format of the vertex in the vertex buffer.
    VkVertexInputBindingDescription binding_description;
    std::vector<VkVertexInputAttributeDescription> attribute_descriptions;
    VkPipelineVertexInputStateCreateInfo vertex_input_info;

    //! @brief Data type of the 3D geometry (typically triangles).
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};

    //! @brief Viewport as in computer graphics pipeline.
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineViewportStateCreateInfo viewport_state;

    //! @brief Rasterization create info.
    VkPipelineRasterizationStateCreateInfo rasterization_state{};

    //! @brief Multisampling create info.
    VkPipelineMultisampleStateCreateInfo multisampling;

    //! @brief Color blend create info.
    std::vector<VkPipelineColorBlendAttachmentState> color_blend_attachments;
    VkPipelineColorBlendStateCreateInfo color_blend;

    std::vector<VkDynamicState> _dynamic_states;
    VkPipelineDynamicStateCreateInfo dynamic_state_info;

    //! @brief Not sure what it is.
    VkPipelineLayoutCreateInfo pipeline_layout_info;

    //! @brief THE BIG FAT CREATE INFO that ties everything together.
    VkGraphicsPipelineCreateInfo pipeline_info;
  };

}  // namespace DO::Kalpana::Vulkan
