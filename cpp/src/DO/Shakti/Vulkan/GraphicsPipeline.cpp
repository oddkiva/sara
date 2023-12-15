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

#include <DO/Shakti/Vulkan/GraphicsPipeline.hpp>


using namespace DO::Kalpana::Vulkan;


auto GraphicsPipeline::Builder::load_shaders() -> void
{
  // Load the compiled shaders.
  SARA_DEBUG << "Load compiled vertex shader...\n";
  vertex_shader = Shakti::Vulkan::read_spirv_compiled_shader(
      vertex_shader_filepath.string());
  SARA_DEBUG << "Creating vertex shader module...\n";
  vertex_shader_module = Shakti::Vulkan::ShaderModule{device, vertex_shader};

  SARA_DEBUG << "Load compiled fragment shader...\n";
  fragment_shader = Shakti::Vulkan::read_spirv_compiled_shader(
      fragment_shader_filepath.string());
  SARA_DEBUG << "Creating fragment shader module...\n";
  fragment_shader_module =
      Shakti::Vulkan::ShaderModule{device, fragment_shader};

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

auto GraphicsPipeline::Builder::initialize_fixed_functions() -> void
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


auto GraphicsPipeline::Builder::create_graphics_pipeline_layout(
    GraphicsPipeline& graphics_pipeline) -> void
{
  // Initialize the graphics pipeline layout.
  SARA_DEBUG << "Initializing the graphics pipeline layout...\n";
  pipeline_layout_info = VkPipelineLayoutCreateInfo{};
  {
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts =
        &static_cast<VkDescriptorSetLayout&>(graphics_pipeline.desc_set_layout);
    pipeline_layout_info.pushConstantRangeCount = 0;
  };
  const auto status = vkCreatePipelineLayout(  //
      device,                                  //
      &pipeline_layout_info,                   //
      nullptr,                                 //
      &graphics_pipeline.pipeline_layout       //
  );
  if (status != VK_SUCCESS)
    throw std::runtime_error{fmt::format(
        "Failed to create the graphics pipeline layout! Error code: {}",
        static_cast<int>(status))};
}

auto GraphicsPipeline::Builder::create_graphics_pipeline(
    GraphicsPipeline& graphics_pipeline) -> void
{
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

    pipeline_info.layout = graphics_pipeline.pipeline_layout;
    pipeline_info.renderPass = render_pass.handle;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;
  };

  const auto status = vkCreateGraphicsPipelines(  //
      device,                                     //
      VK_NULL_HANDLE,                             //
      1,                                          //
      &pipeline_info,                             //
      nullptr,                                    //
      &graphics_pipeline.pipeline                 //
  );
  if (status != VK_SUCCESS)
    throw std::runtime_error{
        fmt::format("Failed to create graphics pipeline! Error code: {}",
                    static_cast<int>(status))};
}

auto GraphicsPipeline::Builder::create() -> GraphicsPipeline
{
  load_shaders();
  initialize_fixed_functions();

  auto graphics_pipeline = GraphicsPipeline{};
  graphics_pipeline.device = device;

  // The list of descriptor sets is what needs to be changed if we create
  // another graphics pipeline.
  graphics_pipeline.desc_set_layout =
      Shakti::Vulkan::DescriptorSetLayout::Builder{device}
          .push_uniform_buffer_layout_binding(0)
          .create();

  create_graphics_pipeline_layout(graphics_pipeline);
  create_graphics_pipeline(graphics_pipeline);

  return graphics_pipeline;
}
