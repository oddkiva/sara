add_subdirectory(Shakti)

add_subdirectory(Sara)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  add_subdirectory(Kalpana)
endif ()
