if (NOT SARA_USE_HALIDE)
  return ()
endif ()
set(BUILD_SHARED_LIBS OFF)

add_subdirectory(Generators)
add_subdirectory(SIFT)
