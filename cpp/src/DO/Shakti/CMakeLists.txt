project(Shakti CXX CUDA)


# Set the version major and minor manually.
set(DO_Shakti_VERSION_MAJOR 0)
set(DO_Shakti_VERSION_MINOR 1)

# List the build options.
option(SHAKTI_BUILD_PYTHON_BINDINGS "Build Python bindings"
  ${SARA_BUILD_PYTHON_BINDINGS})
option(SHAKTI_BUILD_SAMPLES "Build samples in Shakti libraries"
  ${SARA_BUILD_SAMPLES})
option(SHAKTI_BUILD_TESTS "Build unit tests for Shakti libraries"
  ${SARA_BUILD_TESTS})
option(SHAKTI_SELF_CONTAINED_INSTALLATION
  "Install C++ and Python libraries in a single self-contained directory"
  ${SARA_SELF_CONTAINED_INSTALLATION})
option(SHAKTI_BUILD_SHARED_LIBS "Build shared libraries for DO-Sara libraries"
  ${SARA_BUILD_SHARED_LIBS})

# Find required dependencies from Sara.
find_package(DO_Sara COMPONENTS Core REQUIRED)

# Use custom CMake macros.
set(DO_Shakti_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DO_Shakti_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
shakti_dissect_version()

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


# Treat the header-only library.
shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/MultiArray)
shakti_create_common_variables("MultiArray")
add_library(DO_Shakti_MultiArray INTERFACE)
target_sources(DO_Shakti_MultiArray
  INTERFACE "${DO_Shakti_MultiArray_HEADER_FILES}")
target_include_directories(DO_Shakti_MultiArray
  INTERFACE
  $<BUILD_INTERFACE:${DO_Sara_DIR}/cpp/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(DO_Shakti_MultiArray
  INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,AppleClang,Clang>:-Wno-unknown-pragmas>
  $<$<COMPILE_LANG_AND_ID:CUDA,GNU,AppleClang,Clang>:-Xcompiler=-Wno-unknown-pragmas>
  $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4068>
  $<$<COMPILE_LANG_AND_ID:CUDA,MSVC>:-Xcompiler /wd4068>)
add_library(DO::Shakti::MultiArray ALIAS DO_Shakti_MultiArray)

# Create the reamining libraries.
set(DO_Shakti_LIBRARIES ImageProcessing Segmentation Utilities)
foreach (library ${DO_Shakti_LIBRARIES})
  shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/${library})
  shakti_create_common_variables("${library}")
  shakti_generate_library("${library}")
endforeach ()

target_link_libraries(DO_Shakti_ImageProcessing
  PUBLIC
  DO_Shakti_MultiArray
  DO_Shakti_Utilities)
target_link_libraries(DO_Shakti_Segmentation
  PUBLIC
  DO_Shakti_MultiArray
  DO_Shakti_Utilities)
