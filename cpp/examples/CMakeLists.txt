find_package(CUDA 7.0)

add_subdirectory(Kalpana)

add_subdirectory(Sara)

if (NOT CUDA_FOUND)
  sara_message("Cannot use DO::Shakti libraries!")
else ()
  sara_message("Building DO::Shakti Libraries")
  add_subdirectory(Shakti)
endif ()
