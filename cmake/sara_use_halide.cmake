find_package(Halide REQUIRED)


if (NOT SHAKTI_HALIDE_GPU_TARGETS)
  if (APPLE)
    set (SHAKTI_HALIDE_GPU_TARGETS metal)
  elseif (CMAKE_CUDA_COMPILER)
    set (SHAKTI_HALIDE_GPU_TARGETS cuda)
  else ()
    set (SHAKTI_HALIDE_GPU_TARGETS opencl)
  endif ()
endif ()


function (shakti_halide_library _source_filepath)
  get_filename_component(_source_filename ${_source_filepath} NAME_WE)

  add_executable(${_source_filename}.generator ${_source_filepath})
  target_include_directories(${_source_filename}.generator
    PRIVATE
    ${DO_Sara_DIR}/cpp/src
    ${DO_Sara_ThirdParty_DIR}
    ${DO_Sara_ThirdParty_DIR}/eigen)
  target_link_libraries(${_source_filename}.generator PRIVATE Halide::Generator)

  add_halide_library(${_source_filename}
    FROM ${_source_filename}.generator)

  foreach (suffix IN ITEMS ""
                           .generator
                           .runtime
                           .update)
    if (TARGET ${_source_filename}${suffix})
      set_target_properties(${_source_filename}${suffix}
        PROPERTIES
        FOLDER "Libraries/Shakti/Halide/${_source_filename}")
    endif ()
  endforeach ()
endfunction ()


function (shakti_halide_gpu_library _source_filepath)
  get_filename_component(_source_filename ${_source_filepath} NAME_WE)

  add_executable(${_source_filename}.generator ${_source_filepath})
  target_include_directories(${_source_filename}.generator
    PRIVATE
    ${DO_Sara_DIR}/cpp/src
    ${DO_Sara_ThirdParty_DIR}
    ${DO_Sara_ThirdParty_DIR}/eigen)
  target_link_libraries(${_source_filename}.generator PRIVATE Halide::Generator)

  sara_add_halide_library(${_source_filename}
    FROM ${_source_filename}.generator
    TARGETS host
    FEATURES ${SHAKTI_HALIDE_GPU_TARGETS})

  if (APPLE)
    target_link_libraries(${_source_filename}
      INTERFACE "-framework Foundation"
                "-framework Metal")
  endif ()

  foreach (suffix IN ITEMS ""
                           .generator
                           .runtime
                           .update)
    if (TARGET ${_source_filename}${suffix})
      set_target_properties(${_source_filename}${suffix}
        PROPERTIES
        FOLDER "Libraries/Shakti/Halide/${_source_filename}")
    endif ()
  endforeach ()
endfunction ()


function (shakti_halide_library_v2)
  set(_options OPTIONS)
  set(_single_value_args NAME SRCS HALIDE_TARGET)
  set(_multiple_value_args DEPS HALIDE_TARGET_FEATURES)
  cmake_parse_arguments(generator
    "${_options}" "${_single_value_args}" "${_multiple_value_args}" ${ARGN})

  add_executable(${generator_NAME}.generator ${generator_SRCS})
  target_include_directories(${generator_NAME}.generator
    PRIVATE
    ${DO_Sara_DIR}/cpp/src
    ${DO_Sara_ThirdParty_DIR}
    ${DO_Sara_ThirdParty_DIR}/eigen)
  target_link_libraries(${generator_NAME}.generator PRIVATE Halide::Generator)

  if (generator_HALIDE_TARGET_FEATURES)
    add_halide_library(${generator_NAME}
      FROM ${generator_NAME}.generator
      TARGETS host
      FEATURES ${generator_HALIDE_TARGET_FEATURES})
  else ()
    add_halide_library(${generator_NAME}
      FROM ${generator_NAME}.generator
      TARGETS host)
  endif()

  if (APPLE)
    target_link_libraries(${generator_NAME}
      INTERFACE
      "-framework Foundation"
      "-framework Metal")
  endif ()

  foreach (suffix IN ITEMS ""
                           .generator
                           .runtime
                           .update)
    if (TARGET ${generator_NAME}${suffix})
      set_target_properties(${generator_NAME}${suffix}
        PROPERTIES
        FOLDER "Libraries/Shakti/Halide/${generator_NAME}")
    endif ()
  endforeach ()
  target_link_options(${generator_NAME}.runtime INTERFACE
    "$<$<CXX_COMPILER_ID:MSVC>:/ignore:4006>")
  # I want C++17 here.
  target_compile_features(${generator_NAME}.generator
    PRIVATE
    cxx_std_17)
endfunction ()
