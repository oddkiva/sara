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

#include <DO/Sara/Core/DebugUtilities.hpp>

#include <vulkan/vulkan_core.h>

#include <fmt/format.h>

#include <cstdint>
#include <string>
#include <vector>


namespace DO::Shakti::Vulkan {

  auto list_available_instance_layers() -> std::vector<VkLayerProperties>;

  //! Debug hooks to the Vulkan instance
  //! @{
  auto check_validation_layer_support(
      const std::vector<std::string>& requested_validation_layers) -> bool;

  inline VKAPI_ATTR auto VKAPI_CALL
  debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                 VkDebugUtilsMessageTypeFlagsEXT message_type,
                 const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                 void* /* user_data */) -> VkBool32
  {
    std::cerr << fmt::format("[VK-Debug][SEV {:03d}][TYPE {:02d}] {}\n",  //
                             static_cast<std::uint32_t>(message_severity),
                             static_cast<std::uint32_t>(message_type),
                             callback_data->pMessage);
    return VK_FALSE;
  }

  inline auto create_debug_utils_messenger_ext(
      const VkInstance instance,
      const VkDebugUtilsMessengerCreateInfoEXT* create_info,
      const VkAllocationCallbacks* allocator,
      VkDebugUtilsMessengerEXT* debug_messenger) -> VkResult
  {
    // Check if the vendor implemented this function.
    const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func == nullptr)
      return VK_ERROR_EXTENSION_NOT_PRESENT;

    // Then call if if exists.
    return func(instance, create_info, allocator, debug_messenger);
  }

  inline auto destroy_debug_utils_messenger_ext(
      const VkInstance instance, const VkDebugUtilsMessengerEXT debug_messenger,
      const VkAllocationCallbacks* allocator) -> void
  {
    const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func == nullptr)
      return;

    return func(instance, debug_messenger, allocator);
  }
  //! @}

}  // namespace DO::Shakti::Vulkan


namespace DO::Shakti::Vulkan {

  class Instance
  {
  public:
    class Builder;
    friend class Builder;

  public:
    Instance() = default;

    Instance(const Instance&) = delete;

    Instance(Instance&& other)
    {
      swap(other);
    }

    ~Instance()
    {
      if (_debug_messenger != nullptr)
      {
        SARA_DEBUG << fmt::format(
            "[VK] Destroying Vulkan debug messenger: {}...\n",
            fmt::ptr(_debug_messenger));
        destroy_debug_utils_messenger_ext(_instance, _debug_messenger, nullptr);
      }

      if (_instance != nullptr)
      {
        SARA_DEBUG << fmt::format("[VK] Destroying Vulkan instance: {}...\n",
                                  fmt::ptr(_instance));
        vkDestroyInstance(_instance, nullptr);
      }
    }

    auto operator=(const Instance&) -> Instance& = delete;

    auto operator=(Instance&& other) -> Instance&
    {
      swap(other);
      return *this;
    }

    operator VkInstance&()
    {
      return _instance;
    }

    operator VkInstance() const
    {
      return _instance;
    }

    auto swap(Instance& other) -> void
    {
      std::swap(_instance, other._instance);
      std::swap(_debug_messenger, other._debug_messenger);
    }

  private:
    VkInstance _instance = nullptr;
    VkDebugUtilsMessengerEXT _debug_messenger = nullptr;
  };


  //! N.B.: Vulkan follows the factory pattern.
  class Instance::Builder
  {
  public:
    Builder()
    {
      _create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      _create_info.pApplicationInfo = &_app_info;

      // Pre-fill the application metadata.
      _app_info = VkApplicationInfo{};
      _app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      _app_info.applicationVersion = _app_version;
      _app_info.engineVersion = _engine_version;
      _app_info.apiVersion = VK_API_VERSION_1_0;

      // Pre-fill the list of Vulkan extensions.
      _create_info.enabledExtensionCount = 0;
      _create_info.ppEnabledExtensionNames = nullptr;

      // Pre-fill debug create info.
      _debug_create_info.sType =
          VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
      _debug_create_info.messageSeverity =
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      _debug_create_info.messageType =
          VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      _debug_create_info.pfnUserCallback = debug_callback;
      _create_info.enabledLayerCount = 0;
      _create_info.pNext = nullptr;

#if defined(__APPLE__)
      // You need this flags in Apple platforms.
      _create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    }

    auto application_name(const std::string_view& app_name) -> Builder&
    {
      SARA_DEBUG << fmt::format("[VK] Setting app name as: {}\n", app_name);
      _app_name = app_name;
      _app_info.pApplicationName = _app_name.c_str();
      return *this;
    }

    auto engine_name(const std::string_view& engine_name) -> Builder&
    {
      SARA_DEBUG << fmt::format("[VK] Setting engine name as: {}\n",
                                engine_name);
      _engine_name = engine_name;
      _app_info.pEngineName = _engine_name.c_str();
      return *this;
    }

    auto enable_instance_extensions(
        const std::vector<const char*> required_instance_extensions) -> Builder&
    {
      SARA_DEBUG << "[VK] Setting required extensions...\n";
      _required_instance_extensions = required_instance_extensions;

      _create_info.enabledExtensionCount = static_cast<std::uint32_t>(  //
          _required_instance_extensions.size()                          //
      );
      _create_info.ppEnabledExtensionNames =
          _required_instance_extensions.data();

      return *this;
    }

    auto enable_validation_layers(
        const std::vector<const char*>& required_validation_layers) -> Builder&
    {
      SARA_DEBUG << "[VK] Setting required validation layers...\n";
      _required_validation_layers = required_validation_layers;

      if (!_required_validation_layers.empty())
      {
        _create_info.enabledLayerCount = static_cast<uint32_t>(  //
            _required_validation_layers.size()                   //
        );
        _create_info.ppEnabledLayerNames = _required_validation_layers.data();
        _create_info.pNext =
            reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT*>(
                &_debug_create_info);
      }
      else
      {
        _create_info.enabledLayerCount = 0;
        _create_info.pNext = nullptr;
      }

      return *this;
    }

    auto create() -> Instance
    {
      auto instance = Instance{};

      // Finally instantiate the Vulkan instance.
      SARA_DEBUG << "[VK] Initializing a Vulkan instance...\n";
      const auto status =
          vkCreateInstance(&_create_info, nullptr, &instance._instance);
      if (status != VK_SUCCESS)
        throw std::runtime_error{
            fmt::format("Error: failed to create instance! Error code: {}",
                        static_cast<int>(status))  //
        };

      if (!_required_validation_layers.empty())
      {
        create_debug_utils_messenger_ext(instance._instance,
                                         &_debug_create_info, nullptr,
                                         &instance._debug_messenger);
      }

      return instance;
    }

  private:
    VkInstanceCreateInfo _create_info = {};

    //! @brief Vulkan application metadata.
    //! @{
    VkApplicationInfo _app_info = {};
    std::string _app_name;
    std::uint32_t _app_version = VK_MAKE_VERSION(1, 0, 0);
    std::string _engine_name;
    std::uint32_t _engine_version = VK_MAKE_VERSION(1, 0, 0);
    //! @}

    //! @brief Vulkan instance extension.
    std::vector<const char*> _required_instance_extensions;

    //! @brief Vulkan debug hooks.
    VkDebugUtilsMessengerCreateInfoEXT _debug_create_info = {};
    std::vector<const char*> _required_validation_layers;
  };

}  // namespace DO::Shakti::Vulkan
