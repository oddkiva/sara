find_package(DO_Sara COMPONENTS Core REQUIRED)

function(shakti_add_test)
  set(_options OPTIONAL)
  set(_file SOURCES)
  set(_dependencies DEPENDENCIES)
  cmake_parse_arguments(shakti_add_test "${_options}" "${_file}"
                        "${_dependencies}" ${ARGN})

  get_filename_component(_filename "${shakti_add_test_SOURCES}" NAME_WE)
  set(_test_name shakti_${_filename})

  add_executable(${_test_name} ${shakti_add_test_SOURCES})
  target_include_directories(${_test_name} PRIVATE ${CUDA_TOOLKIT_INCLUDE}
                                                   ${Boost_INCLUDE_DIR})
  target_link_libraries(${_test_name} PRIVATE ${Boost_LIBRARIES}
                                              ${shakti_add_test_DEPENDENCIES})

  set_target_properties(${_test_name} PROPERTIES FOLDER "Tests/Shakti")

  target_compile_definitions(${_test_name} PRIVATE -DBOOST_TEST_DYN_LINK)

  target_compile_options(
    ${_test_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  set_target_properties(${_test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR}/bin)

  add_test(NAME ${_test_name} COMMAND $<TARGET_FILE:${_test_name}>)
endfunction()

file(GLOB test_FILES FILES test_*.cpp test_*.cu)

foreach(file ${test_FILES})
  shakti_add_test(SOURCES ${file} DEPENDENCIES
    DO::Sara::Core
    DO::Shakti::Cuda::MultiArray
    DO::Shakti::Cuda::Utilities
    DO::Shakti::Cuda::FeatureDetectors)
endforeach()
