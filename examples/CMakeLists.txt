find_package(CUDA QUIET)

if (CUDA_TOOKIT_INCLUDE AND NOT CUDA_VERSION LESS 7)
  add_subdirectory(Shakti)
endif ()

add_subdirectory(Features)
add_subdirectory(Geometry)
add_subdirectory(Graphics)
add_subdirectory(VideoIO)
