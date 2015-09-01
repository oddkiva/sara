# ==============================================================================
# Debug and verbose functions
#
function (do_message _msg)
  message (STATUS "[DO] ${_msg}")
endfunction ()


function (do_step_message _msg)
  message ("[DO] ${_msg}")
endfunction ()


function (do_substep_message _msg)
  message ("     ${_msg}")
endfunction ()


function (do_list_files _src_files _rel_path _extension)
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
macro (do_dissect_version PROJECT_NAME VERSION)
  # Find version components
  string(REGEX REPLACE "^([0-9]+).*" "\\1"
         ${PROJECT_NAME}_VERSION_MAJOR "${VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1"
         ${PROJECT_NAME}_VERSION_MINOR "${VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1"
         ${PROJECT_NAME}_VERSION_PATCH ${VERSION})
  set(${PROJECT_NAME}_SOVERSION
      "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}")
endmacro ()



# ==============================================================================
# Useful macros to add a new library with minimized effort.
#
macro (do_append_components _component_list _component)
  set(DO_${DO_PROJECT_NAME}_${_component}_USE_FILE UseDO${DO_PROJECT_NAME}${_component})
  list(APPEND "${_component_list}" ${_component})
endmacro ()


macro (do_create_common_variables _library_name)
  set(
    DO_${DO_PROJECT_NAME}_${_library_name}_SOURCE_DIR
    ${DO_${DO_PROJECT_NAME}_SOURCE_DIR}/${_library_name}
    CACHE STRING "Source directory")
  if ("${DO_${DO_PROJECT_NAME}_${_library_name}_SOURCE_FILES}" STREQUAL "")
    set(
      DO_${DO_PROJECT_NAME}_${_library_name}_LIBRARIES ""
      CACHE STRING "Library name")
  else ()
    set(DO_${DO_PROJECT_NAME}_${_library_name}_LIBRARIES
      DO_${DO_PROJECT_NAME}_${_library_name} CACHE STRING "Library name")
  endif ()
endmacro ()


macro (do_include_modules _dep_list)
  foreach (dep ${_dep_list})
    include(${DO_${DO_PROJECT_NAME}_${dep}_USE_FILE})
  endforeach ()
endmacro ()


macro (do_set_internal_dependencies _library_name _dep_list)
  foreach (dep ${_dep_list})
    list(
      APPEND DO_${DO_PROJECT_NAME}_${_library_name}_LINK_LIBRARIES
      ${DO_${DO_PROJECT_NAME}_${dep}_LIBRARIES})
  endforeach ()
endmacro ()


macro (do_append_subdir_files _parentdir _child_dir _hdr_list_var _src_list_var)
  get_filename_component(parentdir_name "${_parentdir}" NAME)

  set(hdr_sublist_var DO_Sara_${parentdir_name}_${_child_dir}_HEADER_FILES)
  set(src_sublist_var DO_Sara_${parentdir_name}_${_child_dir}_SOURCE_FILES)

  file(GLOB ${hdr_sublist_var} FILES ${_parentdir}/${_child_dir}/*.hpp)
  file(GLOB ${src_sublist_var} FILES ${_parentdir}/${_child_dir}/*.cpp)

  source_group("${_child_dir}" FILES
               ${${hdr_sublist_var}} ${${src_sublist_var}})
  list(APPEND ${_hdr_list_var} ${${hdr_sublist_var}})
  list(APPEND ${_src_list_var} ${${src_sublist_var}})

  #message("${hdr_sublist_var} = ${${hdr_sublist_var}}")
endmacro ()


macro(do_glob_directory _curdir)
  #message(STATUS "Parsing current source directory = ${_curdir}")
  file(GLOB curdir_children RELATIVE ${_curdir} ${_curdir}/*)

  get_filename_component(curdir_name "${_curdir}" NAME)
  #message("Directory name: ${curdir_name}")

  file(GLOB DO_Sara_${curdir_name}_HEADER_FILES FILES ${_curdir}/*.hpp)
  file(GLOB DO_Sara_${curdir_name}_SOURCE_FILES FILES ${_curdir}/*.cpp)

  foreach (child ${curdir_children})
    if (IS_DIRECTORY ${_curdir}/${child} AND NOT "${child}" STREQUAL "build")
      #message("Parsing child directory = '${child}'")
      do_append_subdir_files(${_curdir} ${child}
                             DO_Sara_${curdir_name}_HEADER_FILES
                             DO_Sara_${curdir_name}_SOURCE_FILES)
    endif ()
  endforeach ()

  set(DO_Sara_${curdir_name}_MASTER_HEADER ${DO_${DO_PROJECT_NAME}_SOURCE_DIR}/${curdir_name}.hpp)
  source_group("Master Header File" FILES ${DO_Sara_${curdir_name}_MASTER_HEADER})

  list(APPEND DO_Sara_${curdir_name}_HEADER_FILES
       ${DO_Sara_${curdir_name}_MASTER_HEADER})

  #message(STATUS "Master Header:\n ${DO_Sara_${curdir_name}_MASTER_HEADER}")
  #message(STATUS "Header file list:\n ${DO_Sara_${curdir_name}_HEADER_FILES}")
  #message(STATUS "Source file list:\n ${DO_Sara_${curdir_name}_SOURCE_FILES}")
endmacro()


macro (do_append_library _library_name
                         _include_dirs
                         _hdr_files _src_files
                         _lib_dependencies)
  # 1. Verbose comment.
  message(STATUS "[DO] Creating project 'DO_${DO_PROJECT_NAME}_${_library_name}'")

  # 2. Bookmark the project to make sure the library is created only once.
  set_property(GLOBAL PROPERTY _DO_${DO_PROJECT_NAME}_${_library_name}_INCLUDED 1)

  # 3. Include third-party library directories.
  if (NOT "${_include_dirs}" STREQUAL "")
    include_directories(${_include_dirs})
  endif ()

  # 4. Create the project:
  if (NOT "${_src_files}" STREQUAL "")
    # - Case 1: the project contains 'cpp' source files
    #   Specify the source files.
    add_library(DO_${DO_PROJECT_NAME}_${_library_name}
                ${_hdr_files} ${_src_files})

    # Link with other libraries.
    message(STATUS
      "[DO] Linking project 'DO_${DO_PROJECT_NAME}_${_library_name}' with "
      "'${_lib_dependencies}'"
    )
    target_link_libraries(
      DO_${DO_PROJECT_NAME}_${_library_name} ${_lib_dependencies})

    # Form the compiled library output name.
    set(_library_output_basename
        DO_${DO_PROJECT_NAME}_${_library_name}-${DO_${DO_PROJECT_NAME}_VERSION})
    if (DO_BUILD_SHARED_LIBS)
      set (_library_output_name "${_library_output_basename}")
      set (_library_output_name_debug "${_library_output_basename}-d")
    else ()
      set (_library_output_name "${_library_output_basename}-s")
      set (_library_output_name_debug "${_library_output_basename}-sd")
    endif ()

    # Specify output name and version.
    set_target_properties(
      DO_${DO_PROJECT_NAME}_${_library_name}
      PROPERTIES
      VERSION ${DO_${DO_PROJECT_NAME}_VERSION}
      SOVERSION ${DO_${DO_PROJECT_NAME}_SOVERSION}
      OUTPUT_NAME ${_library_output_name}
      OUTPUT_NAME_DEBUG ${_library_output_name_debug})

    # Set correct compile definitions when building the libraries.
    if (DO_BUILD_SHARED_LIBS)
      set(_library_defs "DO_EXPORTS")
    else ()
      set(_library_defs "DO_STATIC")
    endif ()
    set_target_properties(
      DO_${DO_PROJECT_NAME}_${_library_name}
      PROPERTIES
      COMPILE_DEFINITIONS ${_library_defs})

    # Specify where to install the static library.
    install(
      TARGETS DO_${DO_PROJECT_NAME}_${_library_name}
      RUNTIME DESTINATION bin COMPONENT Libraries
      LIBRARY DESTINATION lib/DO/${DO_PROJECT_NAME} COMPONENT Libraries
      ARCHIVE DESTINATION lib/DO/${DO_PROJECT_NAME} COMPONENT Libraries)
  else ()

    # - Case 2: the project is a header-only library
    #   Specify the source files.
    message(STATUS
      "[DO] No linking needed for header-only project "
      "'DO_${DO_PROJECT_NAME}_${_library_name}'")
    add_custom_target(DO_${DO_PROJECT_NAME}_${_library_name} SOURCES ${_hdr_files})
  endif ()

  # 5. Put the library into the folder "DO Libraries".
  set_property(
    TARGET DO_${DO_PROJECT_NAME}_${_library_name} PROPERTY
    FOLDER "DO ${DO_PROJECT_NAME} Libraries")
endmacro ()


macro (do_generate_library _library_name)
  do_append_library(
    ${_library_name}
    "${DO_${DO_PROJECT_NAME}_SOURCE_DIR}"
    "${DO_${DO_PROJECT_NAME}_${_library_name}_HEADER_FILES}"
    "${DO_${DO_PROJECT_NAME}_${_library_name}_SOURCE_FILES}"
    "${DO_${DO_PROJECT_NAME}_${_library_name}_LINK_LIBRARIES}"
  )
endmacro ()



# ==============================================================================
# Specific macro to add a unit test
#
function (do_add_test _test_name _srcs _additional_lib_deps)
  if (POLICY CMP0020)
    cmake_policy(SET CMP0020 OLD)
  endif (POLICY CMP0020)

  # Create a variable containing the list of source files
  set(_srcs_var ${_srcs})

  # Get extra arguments.
  set(extra_macro_args ${ARGN})
  list(LENGTH extra_macro_args num_extra_args)

  # Check if a name is defined for a group of unit tests.
  if (${num_extra_args} GREATER 0)
    list(GET extra_macro_args 0 test_group_name)
  endif ()

  # Create the unit test project
  add_executable(${_test_name} ${_srcs_var})
  target_link_libraries(${_test_name}
                        ${_additional_lib_deps}
                        gtest)

  set_target_properties(
    ${_test_name}
    PROPERTIES
    COMPILE_FLAGS ${DO_DEFINITIONS}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

  add_test(${_test_name}
           "${CMAKE_BINARY_DIR}/bin/${_test_name}")

  if (DEFINED test_group_name)
    set_property(
      TARGET ${_test_name}
      PROPERTY FOLDER "DO ${DO_PROJECT_NAME} Tests/${test_group_name}")
  endif ()
endfunction ()
