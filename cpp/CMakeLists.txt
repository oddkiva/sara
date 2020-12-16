add_subdirectory(third-party)

add_subdirectory(src/DO)

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  return()
endif ()

if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(pipelines)

set(cpp_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(drafts)
