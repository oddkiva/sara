include_directories(${cpp_INCLUDE_DIR})

if (SARA_BUILD_TESTS)
  include_directories(${Boost_INCLUDE_DIR})
  add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

add_subdirectory(ChessboardDetection)
add_subdirectory(Calibration)

if (NOT WIN32)
  add_subdirectory(Compute)
endif ()

add_subdirectory(OpenCL)

add_subdirectory(MatchPropagation)
add_subdirectory(NeuralNetworks)

if (NOT WIN32)
  add_subdirectory(Vulkan)
endif ()

add_subdirectory(NuScenes)

add_subdirectory(Taskflow)

add_subdirectory(Emscripten)
