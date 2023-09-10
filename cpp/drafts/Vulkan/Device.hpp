#pragma once

#include <drafts/Vulkan/PhysicalDevice.hpp>

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <fmt/format.h>


namespace DO::Shakti::Vulkan {

  struct Device
  {
    VkDevice handle = nullptr;

    auto get_device_queue(const std::uint32_t queue_family_index) const
        -> VkQueue
    {
      SARA_DEBUG << fmt::format("[VK] - Fetching the queue from index {}...\n",
                                queue_family_index);
      auto queue = VkQueue{};
      vkGetDeviceQueue(handle, queue_family_index, 0, &queue);
      return queue;
    }
  };

  //! @brief Specify the logical device that we want to create.
  //!
  //! Basically we must bind the capabilities of the physical device to this
  //! logical device.
  struct DeviceCreator
  {
    DeviceCreator(const PhysicalDevice& physical_device)
      : _physical_device{physical_device}
    {
      _create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    }

    //! First and foremost we want the logical device to be able to run graphics
    //! and present operations.
    //!
    //! So We need to bind these family of operations from the physical
    //! device to the logical device.
    auto queue_families(const std::set<std::uint32_t>& queue_family_indices)
        -> DeviceCreator&
    {
      SARA_DEBUG
          << "[VK] - (Basically but not necessarily) checking that the "
             "physical device supports graphics and present operations...\n";

      // Re-populate the list of create infos for each queue family.
      _queue_create_infos.clear();
      for (const auto& queue_family_index : queue_family_indices)
      {

        auto queue_create_info = VkDeviceQueueCreateInfo{};
        queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_create_info.queueFamilyIndex = queue_family_index;
        queue_create_info.queueCount = 1;

        static constexpr auto queue_priority = 1.f;
        queue_create_info.pQueuePriorities = &queue_priority;

        _queue_create_infos.emplace_back(queue_create_info);
      }

      // Rebind the list of queue create infos to the device create info.
      _create_info.queueCreateInfoCount =
          static_cast<std::uint32_t>(_queue_create_infos.size());
      _create_info.pQueueCreateInfos = _queue_create_infos.data();

      return *this;
    }

    //! Bind the validation layers for debugging purposes.
    auto validation_layers(
        const std::vector<const char*>& requested_validation_layers)
        -> DeviceCreator&
    {
      _requested_validation_layers = requested_validation_layers;
      _create_info.enabledLayerCount = _requested_validation_layers.size();
      _create_info.ppEnabledLayerNames = _requested_validation_layers.data();
      return *this;
    }

    //! Bind the Vulkan device extensions we require.
    auto device_extensions(const std::vector<const char*> device_extensions)
        -> DeviceCreator&
    {
      _requested_device_extensions = device_extensions;
      _create_info.enabledExtensionCount = _required_device_extensions.size();
      _create_info.ppEnabledExtensionNames = _required_device_extensions.data();
      return *this;
    }

    //! Bind the Vulkan device features we require.
    auto device_features(const VkPhysicalDeviceFeatures& features = {})
        -> DeviceCreator&
    {
      _physical_device_features = features;
      _create_info.pEnabledFeatures = &_physical_device_features;
      return *this;
    }

    auto create() -> Device
    {
      SARA_DEBUG << "[VK] - Initializing a Vulkan logical device...\n";

      auto device = Device{};
      const auto status = vkCreateDevice(_physical_device, &_create_info,
                                         nullptr, &device.handle);
      if (status != VK_SUCCESS)
      {
        throw std::runtime_error{fmt::format(
            "Error: could not create a logical Vulkan device! Error code: {}",
            static_cast<int>(status))};
      }

      return device;
    }

  private:
    const PhysicalDevice& _physical_device;

    VkDeviceCreateInfo _create_info = {};
    std::vector<VkDeviceQueueCreateInfo> _queue_create_infos;
    std::vector<const char*> _requested_validation_layers;
    std::vector<const char*> _requested_device_extensions;
    VkPhysicalDeviceFeatures _physical_device_features = {};
  };

}  // namespace DO::Shakti::Vulkan
