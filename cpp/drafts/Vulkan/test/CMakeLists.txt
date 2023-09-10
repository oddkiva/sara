file(GLOB test_EasyVulkan_SOURCE_FILES FILES test_*.cpp)

foreach (file ${test_EasyVulkan_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_test(
    NAME ${filename}
    SOURCES ${file}

    DEPENDENCIES EasyVulkan glfw

    FOLDER Vulkan)
endforeach ()
