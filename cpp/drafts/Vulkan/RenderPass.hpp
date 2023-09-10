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

#include <vulkan/vulkan.h>

#include <vector>


namespace DO::Kalpana::Vulkan {

  // The render pass is the render result as in the buffers where we store our
  // result.
  struct RenderPass
  {
    auto create_basic_render_pass(VkDevice device,
                                  const VkFormat swapchain_image_format) -> void
    {
      // 1. Specify the color buffer.
      _color_attachments.resize(1);
      auto& color_attachment = _color_attachments.front();
      color_attachment.format = swapchain_image_format;
      color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
      color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

      // 2. The color buffer is referenced by its index. We take the first one,
      //    i.e. the zero-th one.
      _color_attachment_refs.resize(1);
      auto& color_attachment_ref = _color_attachment_refs.front();
      color_attachment_ref = VkAttachmentReference{
          .attachment = 0,                                    //
          .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //
      };

      // 3. Our render pass is a sequence/tree of subpasses and in our case
      //    we have only one subpass.
      //
      //    In our case, we just want to fill the color buffer with our fragment
      //    shader of a graphics pipeline that we will specify later.
      _subpasses.resize(1);
      auto& subpass = _subpasses.front();
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = _color_attachment_refs.size();
      subpass.pColorAttachments = _color_attachment_refs.data();

      // 4. Link one subpass to one another. Here, if I understood correctly, we
      //    just want the fragment shader to write to the color buffer.
      _dependencies.resize(1);
      auto& dependency = _dependencies.front();
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;
      dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;
      dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

      // 5. Finally initialize a render pass object.
      auto render_pass_create_info = VkRenderPassCreateInfo{};
      render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      render_pass_create_info.attachmentCount = 1;
      render_pass_create_info.pAttachments = &color_attachment;
      render_pass_create_info.subpassCount = _subpasses.size();
      render_pass_create_info.pSubpasses = _subpasses.data();
      render_pass_create_info.dependencyCount = _dependencies.size();
      render_pass_create_info.pDependencies = _dependencies.data();

      if (vkCreateRenderPass(device, &render_pass_create_info, nullptr,
                             &_render_pass) != VK_SUCCESS)
        throw std::runtime_error{"Failed to create render pass!"};
    }

    VkRenderPass _render_pass;
    std::vector<VkAttachmentDescription> _color_attachments;
    std::vector<VkAttachmentReference> _color_attachment_refs;
    std::vector<VkSubpassDescription> _subpasses;
    std::vector<VkSubpassDependency> _dependencies;
  };

}  // namespace vk
