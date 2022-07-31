include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(Sara)

if (CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti Libraries")
  add_subdirectory(Shakti)
else ()
  sara_message("Cannot use DO::Shakti libraries!")
endif ()
