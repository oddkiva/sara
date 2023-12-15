function(shakti_add_test)
  set(_options)
  set(_single_value_args FOLDER)
  set(_multi_value_args SOURCES DEPENDENCIES)
  cmake_parse_arguments(test "${_options}" "${_single_value_args}"
                        "${_multi_value_args}" ${ARGN})

  get_filename_component(_filename "${test_SOURCES}" NAME_WE)
  set(_test_name shakti_${_filename})

  add_executable(${_test_name} ${test_SOURCES})
  target_include_directories(${_test_name} PRIVATE ${CUDA_TOOLKIT_INCLUDE}
                                                   ${Boost_INCLUDE_DIR})
  target_link_libraries(${_test_name} PRIVATE ${Boost_LIBRARIES}
                                              ${test_DEPENDENCIES})

  set_target_properties(${_test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  if(test_FOLDER)
    set_property(TARGET ${_test_name} PROPERTY FOLDER "Tests/Shakti/${test_FOLDER}")
  endif()

  target_compile_definitions(${_test_name} PRIVATE -DBOOST_TEST_DYN_LINK)

  # Disable compile warnings with boost.test. list(APPEND
  # NVCC_DIAG_SUPPRESS_WARNING_IDS "--diag_suppress=611" "--diag_suppress=1388"
  # "--diag_suppress=1394") target_compile_options( ${_test_name} PRIVATE
  # $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:"-Xcudafe
  # ${NVCC_DIAG_SUPPRESS_WARNING_IDS}"> )

  set_target_properties(${_test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR}/bin)

  add_test(NAME ${_test_name} COMMAND $<TARGET_FILE:${_test_name}>)
endfunction()

# Because we use the old CMake way to use CUDA.
include_directories(
  ${CMAKE_SOURCE_DIR}/cpp/src ${CMAKE_SOURCE_DIR}/cpp/third-party
  ${CMAKE_CURRENT_SOURCE_DIR}/../)

add_subdirectory(Cuda)
add_subdirectory(OpenCL)
add_subdirectory(Vulkan)
