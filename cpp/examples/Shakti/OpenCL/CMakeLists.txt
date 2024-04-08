if (NOT OpenCL_FOUND)
  return()
endif ()

add_subdirectory(hello_opencl)
add_subdirectory(image_processing)
