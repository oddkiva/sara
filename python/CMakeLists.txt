find_package(pybind11 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(do/sara/pybind11)
add_subdirectory(do/shakti/pybind11)
