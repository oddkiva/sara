include_directories(${cpp_INCLUDE_DIR})

#add_subdirectory(Core)
add_subdirectory(Compute)

if (USE_HPX)
  add_subdirectory(Parallel)
endif ()
