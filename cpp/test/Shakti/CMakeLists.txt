if (WIN32 AND DO_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()


function (shakti_add_test)
  set(_options OPTIONAL)
  set(_file SOURCES)
  set(_dependencies DEPENDENCIES)
  cmake_parse_arguments(shakti_add_test
    "${_options}" "${_file}" "${_dependencies}" ${ARGN})

  get_filename_component(_filename "${shakti_add_test_SOURCES}" NAME_WE)
  set (_test_name shakti_${_filename})

  add_executable(${_test_name} ${shakti_add_test_SOURCES})
  target_include_directories(${_test_name}
    PRIVATE
    ${CUDA_TOOLKIT_INCLUDE}
    ${Boost_INCLUDE_DIR})
  target_link_libraries(${_test_name}
    PRIVATE
    ${Boost_LIBRARIES}
    ${shakti_add_test_DEPENDENCIES})

  set_target_properties(${_test_name}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    FOLDER "Tests/Shakti")

  target_compile_definitions(${_test_name}
    PRIVATE -DBOOST_TEST_DYN_LINK)

  set_target_properties(${_test_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  add_test(NAME ${_test_name} COMMAND $<TARGET_FILE:${_test_name}>)
endfunction ()

# Because we use the old CMake way to use CUDA.
include_directories(
  ${CMAKE_SOURCE_DIR}/cpp/src
  ${CMAKE_SOURCE_DIR}/cpp/third-party)

add_subdirectory(Utilities)
add_subdirectory(MultiArray)
# add_subdirectory(ImageProcessing)
# add_subdirectory(Segmentation)
