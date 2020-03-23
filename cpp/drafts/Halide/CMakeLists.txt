include(sara_use_halide)

# N.B.: in Mac OS X, the genfiles directory is not formed correctly because of a
# bug in the CMake script in `Halide.cmake` is using the variable
# `CMAKE_CFG_INTDIR` incorrectly.
shakti_halide_library(shakti_halide_rgb_to_gray.cpp)
shakti_halide_library(shakti_halide_gray32f_to_rgb.cpp)
shakti_halide_gpu_library(shakti_halide_gaussian_blur.cpp)


if (SARA_BUILD_SAMPLES)
  add_subdirectory(examples)
endif ()

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()
