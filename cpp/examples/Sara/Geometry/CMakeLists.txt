find_package(DO_Sara COMPONENTS Core Geometry Graphics Visualization REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "Examples/Sara/Geometry")
endmacro()

sara_add_example(polygon_clipping_sutherland_hodgman_example)
sara_add_example(ramer_douglas_peucker_example)

add_subdirectory(raytracer)
