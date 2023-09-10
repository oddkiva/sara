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

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>


namespace DO::Shakti::Vulkan {

  struct PhysicalDevice
  {
    PhysicalDevice() = default;

    PhysicalDevice(const VkPhysicalDevice physical_device)
      : handle{physical_device}
      , queue_families{list_supported_queue_families(physical_device)}
      , extensions_supported{list_supported_extensions(physical_device)}
    {
    }

    static auto list_physical_devices(const VkInstance instance)
        -> std::vector<PhysicalDevice>
    {
      SARA_DEBUG << "  [VK] Counting the number of physical devices...\n";
      auto device_count = std::uint32_t{};
      vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
      SARA_DEBUG << fmt::format("  [VK] Physical device count: {}\n",
                                device_count);

      SARA_DEBUG << "  [VK] Populating the list of physical devices...\n";
      if (device_count == 0)
        return {};

      auto devices = std::vector<VkPhysicalDevice>(device_count);
      vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

      auto devices_transformed = std::vector<PhysicalDevice>(devices.size());
      std::transform(
          devices.begin(), devices.end(), devices_transformed.begin(),
          [](const auto& device) -> PhysicalDevice { return device; });

      return devices_transformed;
    }

    static auto
    list_supported_queue_families(const VkPhysicalDevice& physical_device)
        -> std::vector<VkQueueFamilyProperties>
    {
      auto queue_family_count = std::uint32_t{};
      vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                               &queue_family_count, nullptr);

      auto queue_families =
          std::vector<VkQueueFamilyProperties>(queue_family_count);
      vkGetPhysicalDeviceQueueFamilyProperties(
          physical_device, &queue_family_count, queue_families.data());

      return queue_families;
    }

    static auto
    list_supported_extensions(const VkPhysicalDevice& physical_device)
        -> std::vector<VkExtensionProperties>
    {
      auto extension_count = std::uint32_t{};
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr,
                                           &extension_count, nullptr);

      if (extension_count == 0)
        return {};

      auto extensions = std::vector<VkExtensionProperties>(extension_count);
      vkEnumerateDeviceExtensionProperties(physical_device, nullptr,
                                           &extension_count, extensions.data());
      return extensions;
    }

    auto supports_extension(const std::string_view& extension_name) const
        -> bool
    {
      return std::find_if(
                 extensions_supported.begin(), extensions_supported.end(),
                 [&extension_name](const VkExtensionProperties& extension) {
                   return std::strcmp(extension.extensionName,
                                      extension_name.data()) == 0;
                 }) != extensions_supported.end();
    }

    auto supports_extensions(
        const std::vector<std::string>& extensions_requested) const -> bool
    {
      return std::all_of(
          extensions_requested.begin(), extensions_requested.end(),
          [this](const auto& ext) { return supports_extension(ext); });
    }

    auto supports_queue_family_type(const std::uint32_t queue_family_index,
                                    const VkFlags queue_family_bit_value) const
        -> bool
    {
      const auto& queue_family = queue_families[queue_family_index];
      return (queue_family.queueFlags & queue_family_bit_value) != VkFlags{0};
    }

    auto supports_surface_presentation(const std::uint32_t queue_family_index,
                                       const VkSurfaceKHR surface) const -> bool
    {
      auto present_support = VkBool32{false};
      vkGetPhysicalDeviceSurfaceSupportKHR(handle,              //
                                           queue_family_index,  //
                                           surface,             //
                                           &present_support);
      return static_cast<bool>(present_support);
    }

    operator VkPhysicalDevice&()
    {
      return handle;
    }

    operator VkPhysicalDevice() const
    {
      return handle;
    }

    VkPhysicalDevice handle = nullptr;
    std::vector<VkQueueFamilyProperties> queue_families;
    std::vector<VkExtensionProperties> extensions_supported;
  };

}  // namespace DO::Shakti::Vulkan
