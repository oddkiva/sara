# ==============================================================================
# Debug and verbose functions
#
function (shakti_message _msg)
  message (STATUS "[Shakti] ${_msg}")
endfunction ()


function (shakti_step_message _msg)
  message ("[Shakti] ${_msg}")
endfunction ()


function (shakti_substep_message _msg)
  message ("         ${_msg}")
endfunction ()


function (shakti_list_files _src_files _rel_path _extension)
  file(GLOB _src_files
       RELATIVE ${_rel_path}
       FILES_MATCHING PATTERN ${_extension})

  foreach (l ${LIST})
    set(l ${PATH}/l)
    message (l)
  endforeach ()
  message (${LIST})
endfunction ()



# ==============================================================================
# Useful macros
#
macro (shakti_dissect_version)
  # Retrieve the build number.
  execute_process(
    COMMAND git rev-list --count HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_REV_NUMBER
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set (DO_Shakti_BUILD_NUMBER "${GIT_REV_NUMBER}.r${GIT_COMMIT_HASH}")

  # Build the version.
  set(DO_Shakti_VERSION
    "${DO_Shakti_VERSION_MAJOR}.${DO_Shakti_VERSION_MINOR}.${DO_Shakti_BUILD_NUMBER}")
  set(DO_Shakti_SOVERSION "${DO_Shakti_VERSION_MAJOR}.${DO_Shakti_VERSION_MINOR}")

  configure_file(
    ${DO_Sara_DIR}/cmake/shakti_version.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/shakti_version.cmake @ONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Defines.hpp.in
    ${CMAKE_BINARY_DIR}/cpp/src/DO/Shakti/Defines.hpp @ONLY)

endmacro ()



# ==============================================================================
# Useful macros to add a new library with minimized effort.
#
macro (shakti_append_components _component_list _component)
  set(DO_Shakti_${_component}_USE_FILE UseDOShakti${_component})
  list(APPEND "${_component_list}" ${_component})
endmacro ()


macro (shakti_create_common_variables _library_name)
  set(
    DO_Shakti_${_library_name}_SOURCE_DIR
    ${DO_Shakti_SOURCE_DIR}/Cuda/${_library_name}
    CACHE STRING "Source directory")
  if ("${DO_Shakti_${_library_name}_SOURCE_FILES}" STREQUAL "")
    set(
      DO_Shakti_${_library_name}_LIBRARIES ""
      CACHE STRING "Library name")
  else ()
    set(DO_Shakti_${_library_name}_LIBRARIES
      DO_Shakti_${_library_name} CACHE STRING "Library name")
  endif ()
endmacro ()


macro (shakti_include_modules _dep_list)
  foreach (dep ${_dep_list})
    include(${DO_Shakti_${dep}_USE_FILE})
  endforeach ()
endmacro ()


macro (shakti_set_internal_dependencies _library_name _dep_list)
  foreach (dep ${_dep_list})
    list(
      APPEND DO_Shakti_${_library_name}_LINK_LIBRARIES
      ${DO_Shakti_${dep}_LIBRARIES})
  endforeach ()
endmacro ()


macro (shakti_append_subdir_files  _parentdir _child_dir
                                   _hdr_list_var
                                   _cpp_src_list_var
                                   _cu_src_list_var)
  get_filename_component(parentdir_name "${_parentdir}" NAME)

  file(GLOB ${hdr_sublist_var} FILES ${_parentdir}/${_child_dir}/*.hpp)
  file(GLOB ${cpp_src_sublist_var} FILES ${_parentdir}/${_child_dir}/*.cpp)
  file(GLOB ${cu_src_sublist_var} FILES ${_parentdir}/${_child_dir}/*.cu)

  source_group("${_child_dir}"
    FILES ${${hdr_sublist_var}}
          ${${cpp_src_sublist_var}}
          ${${cu_src_sublist_var}})
  list(APPEND ${_hdr_list_var} ${${hdr_sublist_var}})
  list(APPEND ${_cpp_src_list_var} ${${cpp_src_sublist_var}})
  list(APPEND ${_cu_src_list_var} ${${cu_src_sublist_var}})
endmacro ()


macro(shakti_glob_directory _curdir)
  message(STATUS "Parsing current source directory = ${_curdir}")
  file(GLOB curdir_children RELATIVE ${_curdir} ${_curdir}/*)

  get_filename_component(curdir_name "${_curdir}" NAME)
  message("Directory name: ${curdir_name}")

  file(GLOB DO_Shakti_${curdir_name}_HEADER_FILES FILES ${_curdir}/*.hpp)
  file(GLOB DO_Shakti_${curdir_name}_CPP_FILES FILES ${_curdir}/*.cpp)
  file(GLOB DO_Shakti_${curdir_name}_CU_FILES FILES ${_curdir}/*.cu)

  foreach (child ${curdir_children})
    if (IS_DIRECTORY ${_curdir}/${child})
      message("Parsing child directory = '${child}'")
      shakti_append_subdir_files(${_curdir} ${child}
        DO_Shakti_${curdir_name}_HEADER_FILES
        DO_Shakti_${curdir_name}_CPP_FILES
        DO_Shakti_${curdir_name}_CU_FILES)
    endif ()
  endforeach ()

  set(DO_Shakti_${curdir_name}_MASTER_HEADER ${DO_Shakti_SOURCE_DIR}/Cuda/${curdir_name}.hpp)
  source_group("Master Header File" FILES ${DO_Shakti_${curdir_name}_MASTER_HEADER})

  list(APPEND DO_Shakti_${curdir_name}_HEADER_FILES
       ${DO_Shakti_${curdir_name}_MASTER_HEADER})

  message(STATUS "Master Header:\n ${DO_Shakti_${curdir_name}_MASTER_HEADER}")
  message(STATUS "Header file list:\n ${DO_Shakti_${curdir_name}_HEADER_FILES}")
  message(STATUS "C++ Source file list:\n ${DO_Shakti_${curdir_name}_CPP_FILES}")
  message(STATUS "CUDA Source file list:\n ${DO_Shakti_${curdir_name}_CU_FILES}")
endmacro()


macro (shakti_append_library _library_name
                             _include_dirs
                             _hdr_files _cpp_files _cu_files
                             _lib_dependencies)
  # 1. Verbose comment.
  message(STATUS "[Shakti] Creating project 'DO_Shakti_${_library_name}'")

  # 2. Bookmark the project to make sure the library is created only once.
  set_property(GLOBAL PROPERTY _DO_Shakti_${_library_name}_INCLUDED 1)

  # 3. Create the project:
  # if (SHAKTI_BUILD_SHARED_LIBS)
  #   set(_library_type SHARED)
  # else ()
  #   set(_library_type STATIC)
  # endif ()
  # Force to build static libraries for now: for some reason, it does not work
  # on Windows.
  cuda_add_library(DO_Shakti_${_library_name} STATIC
    ${_hdr_files} ${_cpp_files} ${_cu_files})
  add_library(DO::Shakti::${_library_name} ALIAS DO_Shakti_${_library_name})

  set_target_properties(DO_Shakti_${_library_name}
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CUDA_SEPARABLE_COMPILATION ON)

  # 4. Include third-party library directories.
  if (NOT "${_include_dirs}" STREQUAL "")
    target_include_directories(DO_Shakti_${_library_name}
      PUBLIC
      $<BUILD_INTERFACE:${_include_dirs}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  endif ()


  if (NOT "${_cu_files}${_cpp_files}" STREQUAL "")
    # Link with other libraries.
    message(STATUS
      "[Shakti] Linking project 'DO::Shakti::${_library_name}' with "
      "'${_lib_dependencies}'")

    target_link_libraries(DO_Shakti_${_library_name} ${_lib_dependencies})

    # Form the compiled library output name.
    set(_library_output_basename
        DO_Shakti_${_library_name})
    if (SHAKTI_BUILD_SHARED_LIBS)
      set (_library_output_name "${_library_output_basename}")
      set (_library_output_name_debug "${_library_output_basename}-d")
    else ()
      set (_library_output_name "${_library_output_basename}-s")
      set (_library_output_name_debug "${_library_output_basename}-sd")
    endif ()

    # Specify output name and version.
    set_target_properties(
      DO_Shakti_${_library_name}
      PROPERTIES
      VERSION ${DO_Shakti_VERSION}
      SOVERSION ${DO_Shakti_SOVERSION}
      OUTPUT_NAME ${_library_output_name}
      OUTPUT_NAME_DEBUG ${_library_output_name_debug})

    # Set correct compile definitions when building the libraries.
    # if (SHAKTI_BUILD_SHARED_LIBS)
    #   target_compile_definitions(DO_Shakti_${_library_name}
    #     PRIVATE DO_SHAKTI_EXPORTS)
    # else ()
    target_compile_definitions(DO_Shakti_${_library_name}
      PUBLIC DO_SHAKTI_STATIC)
    # endif ()

    # Specify where to install the static library.
    install(
      TARGETS DO_Shakti_${_library_name}
      RUNTIME DESTINATION bin COMPONENT Libraries
      LIBRARY DESTINATION lib COMPONENT Libraries
      ARCHIVE DESTINATION lib COMPONENT Libraries)
  endif ()

  # 5. Put the library into the folder "DO Shakti Libraries".
  set_property(
    TARGET DO_Shakti_${_library_name} PROPERTY
    FOLDER "Libraries/Shakti")
endmacro ()


macro (shakti_generate_library _library_name)
  shakti_append_library(
    ${_library_name}
    "${DO_Shakti_SOURCE_DIR}/Cuda"
    "${DO_Shakti_${_library_name}_HEADER_FILES}"
    "${DO_Shakti_${_library_name}_CPP_FILES}"
    "${DO_Shakti_${_library_name}_CU_FILES}"
    "${DO_Shakti_${_library_name}_LINK_LIBRARIES}"
  )
endmacro ()


function (shakti_add_example)
  set(_options OPTIONAL)
  set(_executable_name NAME)
  set(_multiple_value_args SOURCES DEPENDENCIES)
  cmake_parse_arguments(example
    "${_options}" "${_executable_name}" "${_multiple_value_args}" ${ARGN})

  message(STATUS "NAME = shakti_${example_NAME}")
  message(STATUS "SOURCES = ${example_SOURCES}")
  message(STATUS "DEPENDENCIES = ${example_DEPENDENCIES}")

  cuda_add_executable(${example_NAME} ${example_SOURCES})
  target_link_libraries(${example_NAME} PRIVATE ${example_DEPENDENCIES})

  set_target_properties(${example_NAME}
    PROPERTIES
    COMPILE_FLAGS ${SARA_DEFINITIONS}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    FOLDER "Examples/Shakti")
endfunction ()
