cmake_minimum_required(VERSION 2.6)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

project(DO_Cuda_Examples)

message(STATUS "  - DO_Cuda_Examples")

find_package(CUDA REQUIRED)
find_package(DO COMPONENTS Core Graphics REQUIRED)

include(${DO_USE_FILE})

include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}
)


macro (do_add_example example)
  cuda_add_executable(${example} ${example}.cu)
  source_group("CUDA Source Files" REGULAR_EXPRESSION ".*\\.cu")
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        COMPILE_DEFINITIONS DO_STATIC)
  target_link_libraries(${example} ${DO_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "CUDA Examples/${example}")
endmacro ()


add_subdirectory(MultiArray)
add_subdirectory(Utilities)

do_add_example(cuda_test)
target_link_libraries(cuda_test DO_Cuda_Utilities DO_Cuda_MultiArray)