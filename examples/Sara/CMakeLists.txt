find_package(CUDA 7.0)

if (CUDA_FOUND)
  message(STATUS "Add Shakti libraries...")
  add_subdirectory(Shakti)
endif ()

add_subdirectory(Features)
add_subdirectory(Geometry)
add_subdirectory(Graphics)
add_subdirectory(VideoIO)
