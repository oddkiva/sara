file(GLOB pyshakti_SRC_FILES FILES *.cpp)

add_library(pyshakti_pybind11 MODULE ${pyshakti_SRC_FILES})
target_include_directories(pyshakti_pybind11 PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(
  pyshakti_pybind11
  PRIVATE pybind11::module
          Halide::Halide
          Halide::Runtime
          DO_Sara_Core
          shakti_rgb8u_to_gray32f_cpu
          shakti_gray32f_to_rgb8u_cpu
          shakti_gaussian_convolution_cpu
          shakti_gaussian_convolution_gpu
          shakti_gradient_2d_32f_gpu_v2
          shakti_polar_gradient_2d_32f_gpu_v2
          shakti_enlarge_cpu
          shakti_enlarge_gpu
          shakti_reduce_32f_cpu
          shakti_reduce_32f_gpu
          shakti_scale_32f_cpu
          shakti_scale_32f_gpu)
# if(CMAKE_CUDA_COMPILER)
#   target_compile_definitions(pyshakti_pybind11
#                              PRIVATE USE_SHAKTI_CUDA_LIBRARIES)
#   target_link_libraries(pyshakti_pybind11 PRIVATE
#     CUDA::cudart
#     DO::Shakti::Cuda::ImageProcessing)
# endif()
set_target_properties(
  pyshakti_pybind11
  PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
             SUFFIX "${PYTHON_MODULE_EXTENSION}"
             FOLDER "Python")

# ============================================================================ #
# Copy the Python bindings in the runtime output directory.
add_custom_target(copy_pyshakti_module)

# Copy pyshakti.[so, dll] to the runtime output directory.
set(pyshakti_LIBRARY_PATH $<TARGET_FILE:pyshakti_pybind11>)
add_custom_command(
  TARGET copy_pyshakti_module
  COMMAND ${CMAKE_COMMAND} -E copy ${pyshakti_LIBRARY_PATH}
          ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(copy_pyshakti_module PROPERTIES FOLDER "Python")

add_dependencies(copy_pyshakti_module pyshakti_pybind11)

# ============================================================================ #
# Install the library in the currently active python environment. if
# (SARA_BUILD_TESTS AND SARA_BUILD_PYTHON_BINDINGS) add_subdirectory(test) endif
# ()

install(
  TARGETS pyshakti_pybind11
  RUNTIME DESTINATION bin COMPONENT Libraries
  LIBRARY DESTINATION lib COMPONENT Libraries
  ARCHIVE DESTINATION lib COMPONENT Libraries)
