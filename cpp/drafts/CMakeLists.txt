include_directories(${cpp_INCLUDE_DIR})

if (NOT WIN32)
  add_subdirectory(Core)
  add_subdirectory(Compute)
  add_subdirectory(OpenCL)
endif ()

if (USE_HPX)
  add_subdirectory(Parallel)
endif ()
