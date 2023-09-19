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

#include <drafts/Vulkan/Swapchain.hpp>

#include <fmt/format.h>


namespace DO::Kalpana::Vulkan {

  // The render pass is the render result as in the buffers where we store our
  // result.
  struct RenderPass
  {
    RenderPass() = default;

    RenderPass(const RenderPass&) = delete;

    RenderPass(RenderPass&& other)
    {
      swap(other);
    }

    ~RenderPass()
    {
      if (device_handle == nullptr)
        return;

      if (handle != nullptr)
      {
        SARA_DEBUG << fmt::format("[VK] Destroying render pass {}\n",
                                  fmt::ptr(handle));
        vkDestroyRenderPass(device_handle, handle, nullptr);
      }
    }

    auto operator=(const RenderPass&) -> RenderPass& = delete;

    auto operator=(RenderPass&& other) -> RenderPass&
    {
      swap(other);
      return *this;
    }

    auto swap(RenderPass& other) -> void
    {
      std::swap(device_handle, other.device_handle);
      std::swap(handle, other.handle);
      color_attachments.swap(other.color_attachments);
      color_attachment_refs.swap(other.color_attachment_refs);
      subpasses.swap(other.subpasses);
      dependencies.swap(other.dependencies);
    }

    auto create_basic_render_pass(const Shakti::Vulkan::Device& device,
                                  const VkFormat swapchain_image_format) -> void
    {
      // First bind the device handle to the render pass object.
      device_handle = device.handle;

      // 1. Specify the color buffer.
      color_attachments.resize(1);
      auto& color_attachment = color_attachments.front();
      color_attachment.format = swapchain_image_format;
      color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
      // Before rendering, clear the previous color values.
      color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      // To be presented to the swapchain.
      color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

      // 2. The color buffer is referenced by its index. We take the first one,
      //    i.e. the zero-th one.
      color_attachment_refs.resize(1);
      auto& color_attachment_ref = color_attachment_refs.front();
      color_attachment_ref = VkAttachmentReference{
          .attachment = 0,                                    //
          .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL  //
      };

      // 3. Our render pass is a sequence/tree of subpasses and in our case
      //    we have only one subpass.
      //
      //    In our case, we just want to fill the color buffer with our fragment
      //    shader of a graphics pipeline that we will specify later.
      subpasses.resize(1);
      auto& subpass = subpasses.front();
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount =
          static_cast<std::uint32_t>(color_attachment_refs.size());
      subpass.pColorAttachments = color_attachment_refs.data();

      // 4. Link one subpass to one another. Here, if I understood correctly, we
      //    just want the fragment shader to write to the color buffer.
      dependencies.resize(1);
      auto& dependency = dependencies.front();
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
      render_pass_create_info.subpassCount =
          static_cast<std::uint32_t>(subpasses.size());
      render_pass_create_info.pSubpasses = subpasses.data();
      render_pass_create_info.dependencyCount =
          static_cast<std::uint32_t>(dependencies.size());
      render_pass_create_info.pDependencies = dependencies.data();

      const auto status = vkCreateRenderPass(
          device.handle, &render_pass_create_info, nullptr, &handle);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("[VK] Failed to create render pass! Error code: {}",
                        static_cast<int>(status))};
    }

    VkDevice device_handle;
    VkRenderPass handle;
    std::vector<VkAttachmentDescription> color_attachments;
    std::vector<VkAttachmentReference> color_attachment_refs;
    std::vector<VkSubpassDescription> subpasses;
    std::vector<VkSubpassDependency> dependencies;
  };

}  // namespace DO::Kalpana::Vulkan
