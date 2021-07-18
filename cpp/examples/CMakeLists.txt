add_subdirectory(Kalpana)

add_subdirectory(Sara)

if (CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti CUDA Libraries")
  add_subdirectory(Shakti)
else ()
  sara_message("Cannot use DO::Shakti CUDA libraries!")
endif ()
