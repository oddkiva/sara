include_directories(${cpp_INCLUDE_DIR})

if (SARA_BUILD_TESTS)
  include_directories(${Boost_INCLUDE_DIR})
  add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

if (NOT WIN32)
  # add_subdirectory(Core)
  add_subdirectory(Compute)
  add_subdirectory(OpenCL)
endif ()

# add_subdirectory(Expression)
add_subdirectory(MatchPropagation)
add_subdirectory(NeuralNetworks)

if (NOT WIN32)
  add_subdirectory(Vulkan)
endif ()

add_subdirectory(NuScenes)

add_subdirectory(Taskflow)
