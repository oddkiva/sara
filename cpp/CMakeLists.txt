add_subdirectory(third-party)

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()

if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()
