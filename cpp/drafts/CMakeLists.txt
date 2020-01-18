include_directories(${cpp_INCLUDE_DIR})

if (SARA_BUILD_TESTS)
  find_package(Boost COMPONENTS
    filesystem serialization system unit_test_framework REQUIRED)
endif ()

if (NOT WIN32)
  # add_subdirectory(Core)
  add_subdirectory(Compute)
  add_subdirectory(OpenCL)
endif ()

add_subdirectory(Expression)
add_subdirectory(NeuralNetworks)

if (SARA_USE_HPX)
  add_subdirectory(Parallel)
endif ()

if (SARA_USE_HALIDE)
  add_subdirectory(Halide)
endif ()
