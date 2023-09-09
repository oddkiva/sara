#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>


namespace vk {

  class Instance
  {
  public:
    Instance()
    {
    }

    ~Instance()
    {
    }

    VkInstance _instance = nullptr;
  };
}  // namespace vk
