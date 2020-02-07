add_subdirectory(Kalpana)

if (NOT CUDA_FOUND)
  sara_message("Cannot use DO::Shakti libraries!")
else ()
  sara_message("Building DO::Shakti Libraries")
  add_subdirectory(Shakti)
endif ()
