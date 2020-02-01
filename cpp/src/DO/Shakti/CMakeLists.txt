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
include(shakti_macros)
shakti_dissect_version()

# Group projects by category.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Create the following libraries.
set(DO_Shakti_LIBRARIES MultiArray ImageProcessing Segmentation Utilities)
foreach (library ${DO_Shakti_LIBRARIES})
  shakti_glob_directory(${DO_Shakti_SOURCE_DIR}/${library})
  shakti_create_common_variables("${library}")
  shakti_generate_library("${library}")
endforeach ()

target_link_libraries(DO_Shakti_ImageProcessing DO::Shakti::Utilities)
target_link_libraries(DO_Shakti_Segmentation DO::Shakti::Utilities)
# target_compile_features(DO_Shakti_Utilities PRIVATE cxx_std_17)
# set_property(TARGET DO_Shakti_Utilities PROPERTY CUDA_STANDARD 14)
