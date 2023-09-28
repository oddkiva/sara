if(NOT SARA_BUILD_TESTS)
  return()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(Sara)
add_subdirectory(Shakti)
