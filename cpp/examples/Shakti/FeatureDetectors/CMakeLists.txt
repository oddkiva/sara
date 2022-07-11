function(shakti_add_example)
  set(_options OPTIONAL)
  set(_file SOURCES)
  set(_dependencies DEPENDENCIES)
  cmake_parse_arguments(shakti_add_example "${_options}" "${_file}"
                        "${_dependencies}" ${ARGN})

  get_filename_component(_filename "${shakti_add_example_SOURCES}" NAME_WE)
  set(name shakti_${_filename})

  add_executable(${name} ${shakti_add_example_SOURCES})
  target_include_directories(${name} PRIVATE ${CUDA_TOOLKIT_INCLUDE}
                                             ${Boost_INCLUDE_DIR}
                                             ${CMAKE_SOURCE_DIR}/cpp)
  target_link_libraries(${name} PRIVATE ${Boost_LIBRARIES}
                                        ${shakti_add_example_DEPENDENCIES})

  set_target_properties(${name} PROPERTIES FOLDER "Examples/Shakti/Cuda")

  #target_compile_options(
  #  ${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_BINARY_DIR}/bin)

  target_link_libraries(${name} PRIVATE CUDA::cudart)
endfunction()

file(GLOB SRC_FILES FILES *.cpp *.cu)

foreach(file ${SRC_FILES})
  shakti_add_example(
    SOURCES
    ${file}
    DEPENDENCIES
    DO::Sara::Core
    DO::Sara::Graphics
    DO::Sara::ImageProcessing
    DO::Sara::VideoIO
    DO::Shakti::Cuda::MultiArray
    DO::Shakti::Cuda::Utilities
    DO::Shakti::Cuda::FeatureDetectors)
endforeach()
