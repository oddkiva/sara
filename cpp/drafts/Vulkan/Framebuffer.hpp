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

#include <drafts/Vulkan/RenderPass.hpp>

namespace DO::Kalpana::Vulkan {

  struct FramebufferSequence
  {
    FramebufferSequence(const Swapchain& swapchain,
                        const RenderPass& render_pass)
      : device{swapchain.device}
    {
      handles.resize(swapchain.image_views.size());

      for (auto i = 0u; i < swapchain.image_views.size(); ++i)
      {
        auto framebuffer_info = VkFramebufferCreateInfo{};
        {
          framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
          framebuffer_info.renderPass = render_pass.handle;
          framebuffer_info.attachmentCount = 1;
          framebuffer_info.pAttachments = &swapchain.image_views[i];
          framebuffer_info.width = swapchain.extent.width;
          framebuffer_info.height = swapchain.extent.height;
          framebuffer_info.layers = 1;
        }

        const auto status = vkCreateFramebuffer(
            swapchain.device.handle, &framebuffer_info, nullptr, &handles[i]);
        if (status != VK_SUCCESS)
          throw std::runtime_error{
              fmt::format("[VK] Failed to create framebuffer! Error code: {}",
                          static_cast<int>(status))};
      }
    }

    ~FramebufferSequence()
    {
      for (const auto fb_handle : handles)
        vkDestroyFramebuffer(device.handle, fb_handle, nullptr);
    }

    const Shakti::Vulkan::Device& device;
    std::vector<VkFramebuffer> handles;
  };

}  // namespace DO::Kalpana::Vulkan
