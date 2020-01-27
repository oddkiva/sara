set(DO_Shakti_VERSION_MAJOR 0)
set(DO_Shakti_VERSION_MINOR 1)

project(DO::Shakti CXX CUDA)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif ()


# ============================================================================ #
# List the build options.
option(SHAKTI_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(SHAKTI_BUILD_SAMPLES "Build samples in Shakti libraries" OFF)
option(SHAKTI_BUILD_TESTS "Build unit tests for Shakti libraries" OFF)
option(SHAKTI_SELF_CONTAINED_INSTALLATION
  "Install C++ and Python libraries in a single self-contained directory" OFF)


# ============================================================================ #
# Find required dependencies from Sara.
find_package(DO_Sara COMPONENTS Core Graphics FeatureDescriptors ImageIO
             REQUIRED)

# Use custom CMake macros.
set(DO_Shakti_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DO_Shakti_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(shakti_macros)
shakti_dissect_version()

# include(shakti_configure_cxx_compiler)
include(shakti_installation_settings)


# ============================================================================ #
# Build the following directories.
#
include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${DO_Sara_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/src
  src)

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# =========================================================================== //
# Build the library.
# Create the following libraries.
set(DO_Shakti_LIBRARIES MultiArray ImageProcessing Segmentation Utilities)
foreach (library ${DO_Shakti_LIBRARIES})
  shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/${library})
  shakti_create_common_variables("${library}")
  shakti_generate_library("${library}")
  target_link_libraries(DO_Shakti_${library}
    ${DO_Sara_Core_LIBRARIES}
    ${CUDA_CUDART_LIBRARY})
endforeach ()


if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Install license files.
  install(FILES ${DO_Shakti_DIR}/COPYING.README
                ${DO_Shakti_DIR}/COPYING.MPL2
          DESTINATION include/DO/Shakti
          COMPONENT Sources)

  # Install CMake scripts.
  configure_file(${DO_Shakti_DIR}/cmake/DO_ShaktiConfig.cmake.in
                 ${CMAKE_BINARY_DIR}/cmake/DO_ShaktiConfig.cmake @ONLY)
  file(GLOB SHAKTI_CMAKE_SCRIPTS ${DO_Shakti_DIR}/cmake/shakti_*.cmake)
  file(COPY ${SHAKTI_CMAKE_SCRIPTS} DESTINATION ${CMAKE_BINARY_DIR}/cmake)
  install(DIRECTORY ${CMAKE_BINARY_DIR}/cmake
          DESTINATION share/DO_Shakti
          COMPONENT Sources)

  # Install source files.
  install(FILES ${CMAKE_BINARY_DIR}/src/DO/Shakti/Defines.hpp
          DESTINATION include/DO/Shakti
          COMPONENT Sources)

  install(DIRECTORY ${DO_Shakti_DIR}/src/DO
          DESTINATION include
          COMPONENT Sources)

  set(CPACK_COMPONENT_Sources_REQUIRED 1)
endif ()


# if (SHAKTI_BUILD_PYTHON_BINDINGS)
#   add_subdirectory(python)
# endif ()
#
# # Compile the C++ unit tests.
# if (SHAKTI_BUILD_TESTS)
#   enable_testing()
#   add_subdirectory(test)
# endif ()
#
# # Compile the sample usage code.
# if (SHAKTI_BUILD_SAMPLES)
#   add_subdirectory(examples)
# endif ()
#
#
# # ============================================================================ #
# # Install the library.
# include(CPack)
