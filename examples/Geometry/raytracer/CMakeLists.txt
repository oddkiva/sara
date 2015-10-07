cmake_minimum_required(VERSION 2.8.7)


find_package(DO_Sara COMPONENTS Core Graphics ImageIO Geometry REQUIRED)

add_executable(
  raytracer_example
  main.cpp
  utilities.hpp
  geometric_objects.hpp geometric_objects.cpp
  raytracer.hpp raytracer.cpp)
target_link_libraries(raytracer_example ${DO_Sara_LIBRARIES})
set_property(TARGET raytracer_example PROPERTY FOLDER "DO Sara Examples/Geometry")
