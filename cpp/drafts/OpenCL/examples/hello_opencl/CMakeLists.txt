file(GLOB KERNEL_SRC_FILES FILES *.cl)
source_group("OpenCL Source Files" FILES ${KERNEL_SRC_FILES})

add_executable(hello_opencl main.cpp ${KERNEL_SRC_FILES})
target_link_libraries(hello_opencl
  PRIVATE
  ${DO_Sara_LIBRARIES}
  DO_Sara_OpenCL_Core)
set_target_properties(hello_opencl PROPERTIES
  COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")
set_property(TARGET hello_opencl PROPERTY FOLDER "Tutorials/OpenCL")
