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
  struct GraphicsPipeline
  {
    struct Builder;

    VkPipelineLayout _pipeline_layout;
    VkPipeline _graphics_pipeline;
  };

  struct GraphicsPipeline::Builder
  {
    Builder(const Shakti::Vulkan::Device& device,
            const Kalpana::Vulkan::RenderPass& render_pass)
      : device{device}
      , render_pass{render_pass}
    {
    }

    auto vertex_shader(const std::filesystem::path& source_filepath) -> Builder&
    {
      vertex_shader_filepath = source_filepath;
      return *this;
    }

    auto fragment_shader(const std::filesystem::path& source_filepath)
        -> Builder&
    {
      fragment_shader_filepath = source_filepath;
      return *this;
    }

    auto compile_shader() -> void
    {
      auto& vertex_shader_stage_info = shader_stage_infos[0];
      vertex_shader_stage_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vertex_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
      vertex_shader_stage_info.module = vertex_shader_module.handle;
      vertex_shader_stage_info.pName = "main";

      auto& fragment_shader_stage_info = shader_stage_infos[1];
      fragment_shader_stage_info.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fragment_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fragment_shader_stage_info.module = fragment_shader_module.handle;
      fragment_shader_stage_info.pName = "main";
    }

    auto create() -> GraphicsPipeline
    {
      compile_shader();

      // 5. Data format of the vertex buffer.
      const auto binding_description = Vertex::get_binding_description();
      const auto attribute_description = Vertex::get_attributes_description();
      vertex_input_info = VkPipelineVertexInputStateCreateInfo{};
      {
        vertex_input_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &binding_description;
        vertex_input_info.vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attribute_description.size());
        vertex_input_info.pVertexAttributeDescriptions =
            attribute_description.data();
      };

      // 6. Data type of the 3D geometry.
      //    Here a list of triangles.
      input_assembly = VkPipelineInputAssemblyStateCreateInfo{};
      {
        input_assembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;
      };

      // Viewport: which portion of the window?
      //
      // Here we want to render on the whole window.
      viewport = VkViewport{
          .x = 0.f,
          .y = 0.f,
          .width = static_cast<float>(image_extent.width),
          .height = static_cast<float>(image_extent.height),
          .minDepth = 0.f,
          .maxDepth = 1.f  //
      };

      scissor = VkRect2D{
          .offset = {0, 0},
          .extent = image_extent  //
      };

      //
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

      rasterizer = VkPipelineRasterizationStateCreateInfo{};
      {
        rasterizer.sType =
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;  //
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.f;
        rasterizer.depthBiasSlopeFactor = 0.f;
        rasterizer.lineWidth = 1.f;
      }

      // Multisampling processing policy.
      // Let's worry about this later.
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
      //
      // 1. Let's worry about this later.
      color_blend_attachment = VkPipelineColorBlendAttachmentState{};
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

      // Let's worry about this later.
      color_blending = VkPipelineColorBlendStateCreateInfo{};
      {
        color_blending.sType =
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.logicOp = VK_LOGIC_OP_COPY;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
        for (auto i = 0; i < 4; ++i)
          color_blending.blendConstants[i] = 0.f;
      };

      // Let's worry about this later.
      pipeline_layout_info = VkPipelineLayoutCreateInfo{};
      {
        pipeline_layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0;
        pipeline_layout_info.pushConstantRangeCount = 0;
      };

      auto graphics_pipeline = GraphicsPipeline{};

      const auto status =
          vkCreatePipelineLayout(device.handle, &pipeline_layout_info, nullptr,
                                 &graphics_pipeline._pipeline_layout);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("Failed to create pipeline layout! Error code: {}",
                        static_cast<int>(status))};

      pipeline_info = VkGraphicsPipelineCreateInfo{};
      {
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        // - Vertex and fragment shaders.
        pipeline_info.stageCount = shader_stage_infos.size();
        pipeline_info.pStages = shader_stage_infos.data();

        // - Vertex buffer data format.
        pipeline_info.pVertexInputState = &vertex_input_info;
        // - We enumerate triangle vertices.
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;

        // The rasterization by the fragment shader.
        pipeline_info.pRasterizationState = &rasterizer;

        // Rendering policy.
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;

        pipeline_info.layout = graphics_pipeline._pipeline_layout;
        pipeline_info.renderPass = render_pass.handle;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_info.basePipelineIndex = -1;
      };

      if (vkCreateGraphicsPipelines(
              device.handle, VK_NULL_HANDLE, 1, &pipeline_info, nullptr,
              &graphics_pipeline._graphics_pipeline) != VK_SUCCESS)
        throw std::runtime_error{"Failed to create graphics pipeline!"};

      return graphics_pipeline;
    }

    const Shakti::Vulkan::Device& device;
    const Kalpana::Vulkan::RenderPass& render_pass;

    //! @brief Paths to shader source.
    std::filesystem::path vertex_shader_filepath;
    std::filesystem::path fragment_shader_filepath;

    //! @brief Compiled shaders.
    Shakti::Vulkan::ShaderModule vertex_shader_module;
    Shakti::Vulkan::ShaderModule fragment_shader_module;

    //! @brief The shader create infos that bind the shader modules.
    std::array<VkPipelineShaderStageCreateInfo, 2> shader_stage_infos;

    //! @brief Data format of the vertex in the vertex buffer.
    VkPipelineVertexInputStateCreateInfo vertex_input_info;

    //! @brief Data type of the 3D geometry (typically triangles).
    VkPipelineInputAssemblyStateCreateInfo input_assembly = {};

    //! @brief Viewport as in computer graphics pipeline.
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineViewportStateCreateInfo viewport_state;

    //! @brief Rasterization create info.
    VkPipelineRasterizationStateCreateInfo rasterizer{};

    //! @brief Multisampling create info.
    VkPipelineMultisampleStateCreateInfo multisampling;

    //! @brief Let's worry about these later.
    VkPipelineColorBlendAttachmentState color_blend_attachment;
    VkPipelineColorBlendStateCreateInfo color_blending;

    //! @brief Not sure what it is.
    VkPipelineLayoutCreateInfo pipeline_layout_info;

    //! @brief THE BIG FAT CREATE INFO that ties everything together.
    VkGraphicsPipelineCreateInfo pipeline_info;
  };

}  // namespace DO::Kalpana::Vulkan
