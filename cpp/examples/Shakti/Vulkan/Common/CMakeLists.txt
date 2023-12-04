add_library(
  SignalHandler #
  HostUniforms.hpp #
  SignalHandler.hpp SignalHandler.cpp)
target_include_directories(SignalHandler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
