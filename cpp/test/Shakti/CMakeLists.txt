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

  cuda_add_executable(${_test_name} ${shakti_add_test_SOURCES})
  target_link_libraries(${_test_name}
    ${Boost_LIBRARIES}
    ${shakti_add_test_DEPENDENCIES})

  target_compile_features(${_test_name} PRIVATE cxx_std_14)
  set_target_properties(${_test_name}
    PROPERTIES
    CUDA_STANDARD 14
    # CUDA_SEPARABLE_COMPILATION ON
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON
    FOLDER "Tests/Shakti")

  target_compile_definitions(${_test_name}
    PRIVATE -DBOOST_TEST_DYN_LINK)

  set_target_properties(${_test_name}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  add_test(NAME ${_test_name} COMMAND $<TARGET_FILE:${_test_name}>)
endfunction ()

add_subdirectory(ImageProcessing)
add_subdirectory(MultiArray)
add_subdirectory(Segmentation)
