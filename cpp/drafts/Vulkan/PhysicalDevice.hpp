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

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>


namespace DO::Shakti::EasyVulkan {

  struct PhysicalDevice
  {
    PhysicalDevice() = default;

    PhysicalDevice(const VkPhysicalDevice physical_device)
      : _physical_device{physical_device}
      , _queue_families{list_supported_queue_families(physical_device)}
      , _extensions_supported{list_supported_extensions(physical_device)}
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
                 _extensions_supported.begin(), _extensions_supported.end(),
                 [&extension_name](const VkExtensionProperties& extension) {
                   return std::strcmp(extension.extensionName,
                                      extension_name.data()) == 0;
                 }) != _extensions_supported.end();
    }

    auto supports_extensions(
        const std::vector<std::string>& extensions_requested) const -> bool
    {
      return std::all_of(
          extensions_requested.begin(), extensions_requested.end(),
          [this](const auto& ext) { return supports_extension(ext); });
    }

    auto supports_queue_family(const VkFlags queue_family_bit_value) const
        -> bool
    {
      return std::find_if(  //
                 _queue_families.begin(), _queue_families.end(),
                 [queue_family_bit_value](
                     const VkQueueFamilyProperties& queue_family) {
                   return (queue_family.queueFlags & queue_family_bit_value) !=
                          VkFlags{0};
                 }) != _queue_families.end();
    }

    auto find_graphics_queue_family_index() -> std::optional<std::uint32_t>
    {
      const auto it = std::find_if(  //
          _queue_families.begin(), _queue_families.end(),
          [](const VkQueueFamilyProperties& queue_family) {
            return (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) !=
                   VkFlags{0};
          });
      if (it == _queue_families.end())
        return std::nullopt;

      return static_cast<std::uint32_t>(it - _queue_families.begin());
    }

    auto find_present_queue_family_index(VkSurface surface)
        -> std::optional<std::uint32_t>
    {
      for (auto i = std::uint32_t{}; i != _queue_families.size(); ++i)
      {
        // Does the physical device have a present queue?
        auto present_support = VkBool32{false};
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface,
                                             &present_support);
        if (present_support)
          return i;
      }

      return std::nullopt;
    }

    operator VkPhysicalDevice&()
    {
      return _physical_device;
    }

    operator VkPhysicalDevice() const
    {
      return _physical_device;
    }

    VkPhysicalDevice _physical_device = nullptr;
    std::vector<VkQueueFamilyProperties> _queue_families;
    std::vector<VkExtensionProperties> _extensions_supported;
  };

}  // namespace DO::Shakti::EasyVulkan
