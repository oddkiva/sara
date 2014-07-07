# ==============================================================================
# Debug and verbose functions
# 
function (do_message _msg)
  message (STATUS "[DO] ${_msg}")
endfunction (do_message _msg)


function (do_step_message _msg)
  message ("[DO] ${_msg}")
endfunction (do_step_message _msg)


function (do_substep_message _msg)
  message ("     ${_msg}")
endfunction (do_substep_message _msg)

function (do_list_files _src_files _rel_path _extension)
  file(GLOB _src_files
       RELATIVE ${_rel_path}
       FILES_MATCHING PATTERN ${_extension})
  foreach (l ${LIST})
    set(l ${PATH}/l)
    message (l)
  endforeach ()
  message (${LIST})
endfunction (do_list_files)


# ==============================================================================
# Useful macros
# 
macro (do_get_os_info)
  string(REGEX MATCH "Linux" OS_IS_LINUX ${CMAKE_SYSTEM_NAME})
  set(DO_LIB_INSTALL_DIR "lib")
  set(DO_INCLUDE_INSTALL_DIR
      "include/DO-${DO_MAJOR_VERSION}.${DO_MINOR_VERSION}")
endmacro (do_get_os_info)


macro (do_dissect_version)
  # Find version components
  string(REGEX REPLACE "^([0-9]+).*" "\\1"
         DO_VERSION_MAJOR "${DO_VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1"
         DO_VERSION_MINOR "${DO_VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1"
         DO_VERSION_PATCH ${DO_VERSION})
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1"
         DO_VERSION_CANDIDATE ${DO_VERSION})
  set(DO_SOVERSION "${DO_VERSION_MAJOR}.${DO_VERSION_MINOR}")
endmacro (do_dissect_version)


# ==============================================================================
# Useful macros to add a new library with minimized effort.
# 
macro (do_append_components _component_list _component)
  set(DO_${_component}_LIBRARIES DO_${_component})
  set(DO_${_component}_USE_FILE UseDO${_component})
  list(APPEND "${_component_list}" ${_component})
endmacro (do_append_components)


macro (do_create_common_variables _library_name)
  set(DO_${_library_name}_SOURCE_DIR ${DO_SOURCE_DIR}/${_library_name})
  set(DO_${_library_name}_LIBRARIES DO_${_library_name})
endmacro ()

macro (do_include_internal_dirs _dep_list)
  foreach (dep ${_dep_list})
    include(${DO_${dep}_USE_FILE})
    message("including ${DO_${dep}_USE_FILE}")
  endforeach ()
endmacro ()

macro (do_set_internal_dependencies _library_name _dep_list)
  foreach (dep ${_dep_list})
    list(APPEND DO_${_library_name}_LINK_LIBRARIES ${DO_${dep}_LIBRARIES})
  endforeach ()
  message ("Dependencies: ${DO_${_library_name}_LINK_LIBRARIES}")
endmacro ()

macro (do_append_subdir_files _parentdir _child_dir _hdr_list_var _src_list_var)
  get_filename_component(parentdir_name "${_parentdir}" NAME)

  set(hdr_sublist_var DO_${parentdir_name}_${_child_dir}_HEADER_FILES)
  set(src_sublist_var DO_${parentdir_name}_${_child_dir}_SOURCE_FILES)

  file(GLOB ${hdr_sublist_var} FILES ${_parentdir}/${_child_dir}/*.hpp)
  file(GLOB ${src_sublist_var} FILES ${_parentdir}/${_child_dir}/*.cpp)
  
  source_group("${_child_dir}" FILES
               ${${hdr_sublist_var}} ${${src_sublist_var}})
  list(APPEND ${_hdr_list_var} ${${hdr_sublist_var}})
  list(APPEND ${_src_list_var} ${${src_sublist_var}})
  
  message("${hdr_sublist_var} = ${${hdr_sublist_var}}")
endmacro ()

macro(do_glob_directory _curdir)
  message(STATUS "Parsing current source directory = ${_curdir}")
  file(GLOB curdir_children RELATIVE ${_curdir} ${_curdir}/*)
  
  get_filename_component(curdir_name "${_curdir}" NAME)
  message("Directory name: ${curdir_name}")
  
  file(GLOB DO_${curdir_name}_HEADER_FILES FILES ${_curdir}/*.hpp)
  file(GLOB DO_${curdir_name}_SOURCE_FILES FILES ${_curdir}/*.cpp)
  
  foreach (child ${curdir_children})
    if (IS_DIRECTORY ${_curdir}/${child} AND NOT "${child}" STREQUAL "build")
      message("Parsing child directory = '${child}'")
      do_append_subdir_files(${_curdir} ${child}
                             DO_${curdir_name}_HEADER_FILES
                             DO_${curdir_name}_SOURCE_FILES)
    endif ()
  endforeach ()
  
  set(DO_${curdir_name}_MASTER_HEADER ${DO_SOURCE_DIR}/${curdir_name}.hpp)
  source_group("Master Header File" FILES ${DO_${curdir_name}_MASTER_HEADER})
  
  list(APPEND DO_${curdir_name}_HEADER_FILES
       ${${DO_${curdir_name}_MASTER_HEADER}})
  
  message(STATUS "Master Header:\n ${DO_${curdir_name}_MASTER_HEADER}")
  message(STATUS "Header file list:\n ${DO_${curdir_name}_HEADER_FILES}")
  message(STATUS "Source file list:\n ${DO_${curdir_name}_SOURCE_FILES}")
endmacro()

macro (do_append_library _library_name
                         _library_type # shared or static
                         _include_dirs
                         _hdr_files _src_files
                         _lib_dependencies)

  # 1. Verbose
  message(STATUS "[DO] Creating project 'DO${_library_name}'")
  # 2. Bookmark the project to make sure we don't try to add the library 
  #    more than once.
  set_property(GLOBAL PROPERTY _DO_${_library_name}_INCLUDED 1)
  # 3. Include third-party library directories.
  if (NOT "${_include_dirs}" STREQUAL "")
    include_directories(${_include_dirs})
  endif ()
  # 4. Create the project:
  if (NOT "${_src_files}" STREQUAL "")
    # - Case 1: the project contains 'cpp' source files
    #   Specify the source files.
    add_library(DO_${_library_name} ${library_type} ${_hdr_files} ${_src_files})
    # Link with external libraries
    message(
      STATUS "[DO] Linking project 'DO_${_library_name}' with "
             "'${_lib_dependencies}'"
    )
    target_link_libraries(DO_${_library_name} ${_lib_dependencies})
  else ()
    # - Case 2: the project is a header-only library
    #   Specify the source files.
    #add_library(DO_${_library_name} STATIC ${_hdr_files})
    message(STATUS 
      "[DO] No linking needed for header-only project 'DO_${_library_name}'")
    #set_target_properties(DO_${_library_name} PROPERTIES LINKER_LANGUAGE CXX)
    add_custom_target(DO_${_library_name} SOURCES ${_hdr_files})
  endif ()
  set_property(TARGET DO_${_library_name} PROPERTY FOLDER "DO Modules")
endmacro (do_append_library)

function (do_set_specific_target_properties _target _additional_compile_flags)
  set(extra_macro_args ${ARGN})
  list(LENGTH extra_macro_args num_extra_args)
  if (${num_extra_args} GREATER 0)
    list(GET extra_macro_args 0 _out_target_name)
  else ()
    set(_out_target_name ${_target})
  endif ()

  set_target_properties(${_target} PROPERTIES
                        VERSION ${DO_VERSION}
                        SOVERSION ${DO_SOVERSION}
                        COMPILE_DEFINITIONS ${_additional_compile_flags}
                        OUTPUT_NAME_DEBUG   ${_out_target_name}-${DO_VERSION}-d
                        OUTPUT_NAME_RELEASE ${_out_target_name}-${DO_VERSION})
endfunction (do_set_specific_target_properties)

macro (do_generate_library _library_name)
  # Static library
  do_append_library(
    ${_library_name} STATIC
    "${DO_SOURCE_DIR}"
    "${DO_${_library_name}_HEADER_FILES}"
    "${DO_${_library_name}_SOURCE_FILES}"
    "${DO_${_library_name}_LINK_LIBRARIES}"
  )
  do_set_specific_target_properties(DO_${_library_name} DO_STATIC)

  # Shared library
  if (DO_BUILD_SHARED_LIBS)
    do_append_library(
      ${_library_name}_SHARED SHARED
      "${DO_SOURCE_DIR}"
      "${DO_${_library_name}_HEADER_FILES}"
      "${DO_${_library_name}_SOURCE_FILES}"
      "${DO_${_library_name}_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_${_library_name}_SHARED 
                                      DO_EXPORTS "DO_${_library_name}")
  endif ()
endmacro ()


macro (do_add_msvc_precompiled_header _pch _src_var)
  # Get extra arguments.
  set(extra_macro_args ${ARGN})
  list(LENGTH extra_macro_args num_extra_args)
  
  if (MSVC)
    if (${num_extra_args} GREATER 0)
      list(GET extra_macro_args 0 _pch_basename)
    else ()
      get_filename_component(_pch_basename ${_pch} NAME_WE)
    endif ()

    set(_pch_binary "${CMAKE_CURRENT_BINARY_DIR}/${_pch_basename}.pch")
    set(_srcs ${${_src_var}})

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set(_pch_src ${CMAKE_CURRENT_BINARY_DIR}/${_pch_basename}.cpp)
    file(WRITE ${_pch_src} "// Precompiled header unity generated by CMake\n")
    file(APPEND ${_pch_src} "#include \"${_pch}\"\n")

    set_source_files_properties(${_pch_src} PROPERTIES
      COMPILE_FLAGS "/Yc\"${_pch}\" /Fp\"${_pch_binary}\""
      OBJECT_OUTPUTS "${_pch_binary}")
    set_source_files_properties(${_srcs} PROPERTIES
      COMPILE_FLAGS "/Yu\"${_pch}\" /FI\"${_pch_binary}\" /Fp\"${_pch_binary}\""
      OBJECT_DEPENDS "${_pch_binary}")
    list(APPEND ${_src_var} ${_pch_src})
  endif ()
endmacro (do_add_msvc_precompiled_header)


# ==============================================================================
# Specific macro to add a unit test
#
function (do_test _test_name _srcs _additional_lib_deps)
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

  # Check if we want to use precompiled header.
  if (${num_extra_args} GREATER 1)
    list(GET extra_macro_args -1 pch)
  endif ()

  if (DEFINED pch)
    do_substep_message(
      "Activating precompiled header: '${pch}' for unit test: "
      "'DO_${_test_name}_test'"
    )
    list(APPEND _srcs_var ${pch})
    do_add_msvc_precompiled_header(${pch} _srcs_var ${_test_name}_test_PCH)
  endif ()

  # Create the unit test project
  include_directories(${gtest_DIR}/include)
  add_executable(${_test_name} ${_srcs_var})
  target_link_libraries(${_test_name}
                        ${_additional_lib_deps}
                        gtest)
  set_target_properties(
    ${_test_name} PROPERTIES
    COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}"
    COMPILE_DEFINITIONS DO_STATIC)
  add_test(${_test_name}
           "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_test_name}")
  
  if (DEFINED test_group_name)
    set_property(TARGET ${_test_name}
                 PROPERTY FOLDER "DO Tests/${test_group_name}")
  endif ()
endfunction (do_test)
