if (NOT SARA_USE_HALIDE)
  return ()
endif ()
add_subdirectory(Components)
add_subdirectory(Generators)
add_subdirectory(SIFT)
