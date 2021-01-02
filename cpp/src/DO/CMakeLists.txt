add_subdirectory(Sara)

if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  add_subdirectory(Kalpana)
endif ()

if (CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti Libraries")
  add_subdirectory(Shakti)
else ()
  sara_message("Cannot use DO::Shakti libraries!")
endif ()
