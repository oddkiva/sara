find_package(HPX REQUIRED)

add_hpx_executable(hello_world
  SOURCES hello_world.cpp
  COMPONENT_DEPENDENCIES iostreams)
