add_executable(nuimages_cpp_devkit_usage_example
               nuimages_cpp_devkit_usage_example.cpp)
target_link_libraries(
  nuimages_cpp_devkit_usage_example
  PRIVATE DO::Sara::NuScenes #
          DO::Sara::Core #
          DO::Sara::Graphics #
          DO::Sara::ImageIO)
set_property(TARGET nuimages_cpp_devkit_usage_example
             PROPERTY FOLDER "Datasets/NuScenes")

add_executable(nuscenes_cpp_devkit_usage_example
               nuscenes_cpp_devkit_usage_example.cpp)
target_link_libraries(
  nuscenes_cpp_devkit_usage_example
  PRIVATE DO::Sara::NuScenes #
          DO::Sara::Core #
          DO::Sara::Graphics #
          DO::Sara::ImageIO)
set_property(TARGET nuscenes_cpp_devkit_usage_example
             PROPERTY FOLDER "Datasets/NuScenes")
