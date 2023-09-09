#pragma once

#include "Shader.hpp"

#include <array>
#include <filesystem>
#include <vulkan/vulkan_core.h>


namespace vk {

  //! The graphics pipeline, which is called the render pipeline in Metal.
  //!
  //! This object specifies:
  //! - what vertex shader
  //! - what fragment shader
  //! we want to use.
  struct GraphicsPipeline
  {
    auto create_render_pipeline(const VkDevice device,
                                const VkRenderPass render_pass,
                                const std::string& program_path,
                                const VkExtent2D& image_extent) -> void
    {
      // 1. Where do the vertex and fragment shader code lives?
      namespace fs = std::filesystem;
      const auto p = fs::path{program_path};

      // 2. Retrieve the vertex shader code.
      const auto vertex_shader_code =
          read_shader_file((p / "vert.spv").string());
      const auto [vertex_shader_module, vshader_compiled] =
          create_shader_module(vertex_shader_code);
      auto vertex_shader_stage_info = VkPipelineShaderStageCreateInfo{};
      {
        vertex_shader_stage_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertex_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertex_shader_stage_info.module = vertex_shader_module;
        vertex_shader_stage_info.pName = "main";
      }

      // 3. Retrieve the fragment shader code.
      const auto fragment_shader_code =
          read_shader_file((p / "frag.spv").string());
      const auto [fragment_shader_module, fshader_compiled] =
          create_shader_module(fragment_shader_code);
      auto fragment_shader_stage_info = VkPipelineShaderStageCreateInfo{};
      {
        fragment_shader_stage_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragment_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragment_shader_stage_info.module = fragment_shader_module;
        fragment_shader_stage_info.pName = "main";
      }

      // 4. The render pipeline typically transforms the 3D geometry.
      auto shader_stages = std::array{
          // Vertex shader which typically does the following:
          // 1. 3D vertex coords -> modelview coordinates
          // 2. modelview coordinates -> projection coordinates
          // 3. projection coordinates -> clip coordinates
          vertex_shader_stage_info,
          // 4. Rasterization by fragment shader
          fragment_shader_stage_info  //
      };

      // 5. Data format of the vertex buffer.
      const auto binding_description = Vertex::get_binding_description();
      const auto attribute_description = Vertex::get_attributes_description();

      auto vertex_input_info = VkPipelineVertexInputStateCreateInfo{};
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
      auto input_assembly = VkPipelineInputAssemblyStateCreateInfo{};
      {
        input_assembly.sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;
      };

      // Viewport: which portion of the window?
      //
      // Here we want to render on the whole window.
      const auto viewport = VkViewport{
          .x = 0.f,
          .y = 0.f,
          .width = static_cast<float>(image_extent.width),
          .height = static_cast<float>(image_extent.height),
          .minDepth = 0.f,
          .maxDepth = 1.f  //
      };

      const auto scissor = VkRect2D{
          .offset = {0, 0},
          .extent = image_extent  //
      };

      //
      auto viewport_state = VkPipelineViewportStateCreateInfo{};
      {
        viewport_state.sType =
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.pNext = nullptr;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;
      };

      auto rasterizer = VkPipelineRasterizationStateCreateInfo{};
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
      auto multisampling = VkPipelineMultisampleStateCreateInfo{};
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
      auto color_blend_attachment = VkPipelineColorBlendAttachmentState{};
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
      auto color_blending = VkPipelineColorBlendStateCreateInfo{};
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
      auto pipeline_layout_info = VkPipelineLayoutCreateInfo{};
      {
        pipeline_layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 0;
        pipeline_layout_info.pushConstantRangeCount = 0;
      };

      if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                                 &_pipeline_layout) != VK_SUCCESS)
        throw std::runtime_error{"Failed to create pipeline layout!"};

      auto pipeline_info = VkGraphicsPipelineCreateInfo{};
      {
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

        // - Vertex and fragment shaders.
        pipeline_info.stageCount = shader_stages.size();
        pipeline_info.pStages = shader_stages.data();

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

        pipeline_info.layout = _pipeline_layout;
        pipeline_info.renderPass = render_pass;
        pipeline_info.subpass = 0;
        pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
        pipeline_info.basePipelineIndex = -1;
      };

      if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                    nullptr, &_graphics_pipeline) != VK_SUCCESS)
        throw std::runtime_error{"Failed to create graphics pipeline!"};

      // Clean up the shaders.
      vkDestroyShaderModule(device, vertex_shader_module, nullptr);
      vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    }

    VkPipelineLayout _pipeline_layout;
    VkPipeline _graphics_pipeline;
  };


}  // namespace vk
