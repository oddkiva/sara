find_package(DO_Sara COMPONENTS Core Graphics ImageIO REQUIRED)

add_library(NuScenes
  NuImages.hpp NuImages.cpp
  NuScenes.hpp NuScenes.cpp)
target_compile_features(NuScenes
  PUBLIC
  cxx_std_17)
target_include_directories(NuScenes
  PUBLIC
  ${cpp_INCLUDE_DIR}/third-party
  ${cpp_INCLUDE_DIR}/third-party/eigen)
target_link_libraries(NuScenes
  PRIVATE
  Boost::filesystem)

add_executable(nuimages_cpp_devkit_usage_example
  nuimages_cpp_devkit_usage_example.cpp)
target_link_libraries(nuimages_cpp_devkit_usage_example
  PRIVATE
  NuScenes
  ${DO_Sara_LIBRARIES})

add_executable(nuscenes_cpp_devkit_usage_example nuscenes_cpp_devkit_usage_example.cpp)
target_link_libraries(nuscenes_cpp_devkit_usage_example
  PRIVATE
  NuScenes
  ${DO_Sara_LIBRARIES})
