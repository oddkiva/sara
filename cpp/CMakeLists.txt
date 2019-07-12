add_subdirectory(third-party)

add_subdirectory(src/DO)

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()

if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

add_subdirectory(pipelines)

add_subdirectory(drafts)
