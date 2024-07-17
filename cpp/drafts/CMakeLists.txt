if(NOT SARA_BUILD_DRAFTS)
  return()
endif()

include_directories(${CMAKE_SOURCE_DIR}/cpp)

if(SARA_BUILD_TESTS)
  include_directories(${Boost_INCLUDE_DIR})
  add_definitions(-DBOOST_ALL_NO_LIB)
endif()

add_subdirectory(Compute)
add_subdirectory(MatchPropagation)
add_subdirectory(Taskflow)

# Disable for now because the ONNX runtime library was compiled with CUDA 11 and
# we use CUDA 12. add_subdirectory(OnnxRuntime)
