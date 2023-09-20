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

#include <drafts/Vulkan/Device.hpp>
#include <drafts/Vulkan/RenderPass.hpp>
#include <drafts/Vulkan/Shader.hpp>

#include <array>
#include <filesystem>
#include <vulkan/vulkan_core.h>


namespace DO::Kalpana::Vulkan {

  //! The graphics pipeline, which is called the render pipeline in Metal.
  //!
  //! This object specifies:
  //! - what vertex shader
  //! - what fragment shader
  //! we want to use.
  class GraphicsPipeline
  {
  public:
    struct Builder;
    friend struct Builder;

  public:
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
      if (_device == nullptr)
        return;

      if (_pipeline_layout != nullptr)
      {
        SARA_DEBUG << "Destroying graphics pipeline layout...\n";
        vkDestroyPipelineLayout(_device, _pipeline_layout, nullptr);
      }

      if (_pipeline != nullptr)
      {
        SARA_DEBUG << "Destroying graphics pipeline...\n";
        vkDestroyPipeline(_device, _pipeline, nullptr);
      }
    }

    auto device() const -> VkDevice
    {
      return _device;
    }

    auto pipeline_layout() const -> VkPipelineLayout
    {
      return _pipeline_layout;
    }

    operator VkPipeline() const
    {
      return _pipeline;
    }

    auto swap(GraphicsPipeline& other) -> void
    {
      std::swap(_device, other._device);
      std::swap(_pipeline_layout, other._pipeline_layout);
      std::swap(_pipeline, other._pipeline);
    }

  private:
    VkDevice _device = nullptr;
    VkPipelineLayout _pipeline_layout = nullptr;
    VkPipeline _pipeline = nullptr;
  };

  struct GraphicsPipeline::Builder
  {
    Builder(const Shakti::Vulkan::Device& device,
            const Kalpana::Vulkan::RenderPass& render_pass)
      : device{device}
      , render_pass{render_pass}
    {
    }

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

    auto create() -> GraphicsPipeline
    {
      load_shaders();
      initialize_fixed_functions();

      auto graphics_pipeline = GraphicsPipeline{};

      graphics_pipeline._device = device.handle;

      // Initialize the graphics pipeline layout.
      SARA_DEBUG << "Initializing the graphics pipeline layout...\n";
      pipeline_layout_info = VkPipelineLayoutCreateInfo{};
      {
        pipeline_layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0;
        pipeline_layout_info.pushConstantRangeCount = 0;
      };
      auto status = vkCreatePipelineLayout(    //
          device.handle,                       //
          &pipeline_layout_info,               //
          nullptr,                             //
          &graphics_pipeline._pipeline_layout  //
      );
      if (status != VK_SUCCESS)
        throw std::runtime_error{fmt::format(
            "Failed to create the graphics pipeline layout! Error code: {}",
            static_cast<int>(status))};

      // Initialize the graphics pipeline.
      SARA_DEBUG << "Initializing the graphics pipeline...\n";
      pipeline_info = {};
      {
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        // - Vertex and fragment shaders.
        pipeline_info.stageCount =
            static_cast<std::uint32_t>(shader_stage_infos.size());
        pipeline_info.pStages = shader_stage_infos.data();

        // - Vertex buffer data format.
        pipeline_info.pVertexInputState = &vertex_input_info;
        // - We enumerate triangle vertices.
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;

        // The rasterization state.
        pipeline_info.pRasterizationState = &rasterization_state;

        // Rendering policy.
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blend;

        pipeline_info.layout = graphics_pipeline._pipeline_layout;
        pipeline_info.renderPass = render_pass.handle;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_info.basePipelineIndex = -1;
      };

      status = vkCreateGraphicsPipelines(  //
          device.handle,                   //
          VK_NULL_HANDLE,                  //
          1,                               //
          &pipeline_info,                  //
          nullptr,                         //
          &graphics_pipeline._pipeline     //
      );
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("Failed to create graphics pipeline! Error code: {}",
                        static_cast<int>(status))};

      return graphics_pipeline;
    }

    auto vertex_shader_stage_info() -> VkPipelineShaderStageCreateInfo&
    {
      return shader_stage_infos[0];
    }

    auto fragment_shader_stage_info() -> VkPipelineShaderStageCreateInfo&
    {
      return shader_stage_infos[1];
    }

  private:
    auto load_shaders() -> void
    {
      // Load the compiled shaders.
      SARA_DEBUG << "Load compiled vertex shader...\n";
      vertex_shader = Shakti::Vulkan::read_spirv_compiled_shader(
          vertex_shader_filepath.string());
      SARA_DEBUG << "Creating vertex shader module...\n";
      vertex_shader_module =
          Shakti::Vulkan::ShaderModule{device.handle, vertex_shader};

      SARA_DEBUG << "Load compiled fragment shader...\n";
      fragment_shader = Shakti::Vulkan::read_spirv_compiled_shader(
          fragment_shader_filepath.string());
      SARA_DEBUG << "Creating fragment shader module...\n";
      fragment_shader_module =
          Shakti::Vulkan::ShaderModule{device.handle, fragment_shader};

      // Rebind the shader module references to their respective shader stage
      // infos.
      SARA_DEBUG << "Rebind vertex shader module to vertex shader stage create "
                    "info...\n";
      auto& vssi = vertex_shader_stage_info();
      vssi = {};  // Very important.
      vssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vssi.stage = VK_SHADER_STAGE_VERTEX_BIT;
      vssi.module = vertex_shader_module.handle;
      vssi.pName = "main";

      SARA_DEBUG << "Rebind fragment shader module to fragment shader stage "
                    "create info...\n";
      auto& fssi = fragment_shader_stage_info();
      fssi = {};  // Very important.
      fssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fssi.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fssi.module = fragment_shader_module.handle;
      fssi.pName = "main";
    }

    auto initialize_fixed_functions() -> void
    {
      SARA_DEBUG << "Initialize the viewport state create info...\n";
      viewport_state = VkPipelineViewportStateCreateInfo{};
      {
        viewport_state.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.pNext = nullptr;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;
      };

      SARA_DEBUG << "Initialize the rasterization state create info...\n";
      rasterization_state = VkPipelineRasterizationStateCreateInfo{};
      {
        rasterization_state.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization_state.depthClampEnable = VK_FALSE;
        rasterization_state.rasterizerDiscardEnable = VK_FALSE;  //
        rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization_state.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterization_state.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterization_state.depthBiasEnable = VK_FALSE;
        rasterization_state.depthBiasConstantFactor = 0.f;
        rasterization_state.depthBiasSlopeFactor = 0.f;
        rasterization_state.lineWidth = 1.f;
      }

      // Multisampling processing policy.
      SARA_DEBUG << "Initialize the multisampling state create info...\n";
      multisampling = VkPipelineMultisampleStateCreateInfo{};
      {
        multisampling.sType =
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.minSampleShading = 1.f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
      }

      // Color blending policy.
      SARA_DEBUG << "Initialize the color blend attachment state...\n";
      color_blend_attachments.resize(1);
      auto& color_blend_attachment = color_blend_attachments.front();
      {
        color_blend_attachment.blendEnable = VK_FALSE;
        color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;  //
        color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |  //
                                                VK_COLOR_COMPONENT_G_BIT |  //
                                                VK_COLOR_COMPONENT_B_BIT |  //
                                                VK_COLOR_COMPONENT_A_BIT;   //
      }

      SARA_DEBUG << "Initialize the color blend state create info...\n";
      color_blend = VkPipelineColorBlendStateCreateInfo{};
      {
        color_blend.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blend.logicOpEnable = VK_FALSE;
        color_blend.logicOp = VK_LOGIC_OP_COPY;
        color_blend.attachmentCount =
            static_cast<std::uint32_t>(color_blend_attachments.size());
        color_blend.pAttachments = color_blend_attachments.data();
        for (auto i = 0; i < 4; ++i)
          color_blend.blendConstants[i] = 0.f;
      };
    }

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
    std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions;
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

    //! @brief Not sure what it is.
    VkPipelineLayoutCreateInfo pipeline_layout_info;

    //! @brief THE BIG FAT CREATE INFO that ties everything together.
    VkGraphicsPipelineCreateInfo pipeline_info;
  };

}  // namespace DO::Kalpana::Vulkan
