add_library(
  DO_Sara_NuScenes #
  STATIC #
  NuImages.hpp NuImages.cpp #
  NuScenes.hpp NuScenes.cpp)
add_library(DO::Sara::NuScenes ALIAS DO_Sara_NuScenes)
target_include_directories(
  DO_Sara_NuScenes #
  PUBLIC ${CMAKE_SOURCE_DIR}/cpp/src #
         ${CMAKE_SOURCE_DIR}/cpp/third-party)
target_link_libraries(DO_Sara_NuScenes PUBLIC Eigen3::Eigen)
set_property(TARGET DO_Sara_NuScenes PROPERTY FOLDER "Sara/Datasets/NuScenes")
