if(NOT Vulkan_FOUND)
  return()
endif()

file(GLOB DO_Shakti_Vulkan_SOURCE_FILES FILES *.cpp *.hpp)

add_library(DO_Shakti_Vulkan ${DO_Shakti_Vulkan_SOURCE_FILES})
add_library(DO::Shakti::Vulkan ALIAS DO_Shakti_Vulkan)

target_link_libraries(
  DO_Shakti_Vulkan
  PUBLIC DO::Sara::Core #
         Vulkan::Vulkan #
         glfw)

set_target_properties(DO_Shakti_Vulkan PROPERTIES FOLDER "Libraries/Shakti")
