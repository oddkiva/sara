include_directories(${cpp_INCLUDE_DIR})

#add_subdirectory(Core)
add_subdirectory(Compute)
add_subdirectory(OpenCL)

if (USE_HPX)
  add_subdirectory(Parallel)
endif ()
