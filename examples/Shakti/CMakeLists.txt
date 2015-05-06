cmake_minimum_required(VERSION 2.6)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()

project(DO_Shakti_Examples)

message(STATUS "  - DO_Shakti_Examples")

find_package(CUDA 7.0 REQUIRED)
find_package(DO COMPONENTS Core Graphics REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)

include(${DO_USE_FILE})

include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}
)


macro (do_add_example example)
  cuda_add_library(${example} ${example}.cu)
  source_group("CUDA Source Files" REGULAR_EXPRESSION ".*\\.cu$")
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
                        COMPILE_DEFINITIONS DO_STATIC)
  target_link_libraries(${example} ${DO_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "CUDA Examples")
endmacro ()


function (cuda_add_example)
   # Get the test executable name.
   list(GET ARGN 0 EXAMPLE_NAME)
   message(STATUS "EXAMPLE NAME = ${EXAMPLE_NAME}")

   # Get the list of source files.
   list(REMOVE_ITEM ARGN ${EXAMPLE_NAME})
   message(STATUS "SOURCE FILES = ${ARGN}")

   # Split the list of source files in two sub-lists:
   # - list of CUDA source files.
   # - list of regular C++ source files.
   set (CUDA_SOURCE_FILES "")
   set (CPP_SOURCE_FILES "")
   foreach (SOURCE ${ARGN})
     if (${SOURCE} MATCHES "(.*).cu$")
       list(APPEND CUDA_SOURCE_FILES ${SOURCE})
     else ()
       list(APPEND CPP_SOURCE_FILES ${SOURCE})
     endif()
   endforeach ()
   message(STATUS "CUDA_SOURCE_FILES = ${CUDA_SOURCE_FILES}")
   message(STATUS "CPP_SOURCE_FILES = ${CPP_SOURCE_FILES}")

   # Add the C++ test executable.
   add_executable(${EXAMPLE_NAME} ${CPP_SOURCE_FILES})
   set_property(TARGET ${EXAMPLE_NAME} PROPERTY FOLDER "CUDA Examples")
   set_target_properties(
     ${EXAMPLE_NAME} PROPERTIES
     COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
     COMPILE_DEFINITIONS DO_STATIC
   )

   # Create an auxilliary library for CUDA based code.
   # This is a workaround to do unit-test CUDA code with gtest.
   if (NOT "${CUDA_SOURCE_FILES}" STREQUAL "")
     source_group("CUDA Source Files" REGULAR_EXPRESSION ".*\\.cu$")
     cuda_add_library(${EXAMPLE_NAME}_CUDA_AUX ${CUDA_SOURCE_FILES})
     target_link_libraries(${EXAMPLE_NAME}_CUDA_AUX
                           DO_Shakti_Utilities
                           ${DO_LIBRARIES})
     # Group the unit test in the "Tests" folder.
     set_property(
       TARGET ${EXAMPLE_NAME}_CUDA_AUX PROPERTY FOLDER "CUDA Examples")

     target_link_libraries(${EXAMPLE_NAME} ${EXAMPLE_NAME}_CUDA_AUX ${DO_LIBRARIES})
   endif ()
endfunction ()


add_subdirectory(MultiArray)
add_subdirectory(Utilities)

file(GLOB cuda_test_SRC_FILES FILES *.cu *.cpp *.hpp)
cuda_add_example(cuda_test ${cuda_test_SRC_FILES})
target_link_libraries(cuda_test DO_Shakti_Utilities DO_Shakti_MultiArray)
