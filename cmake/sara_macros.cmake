# ==============================================================================
# Debug and verbose functions
#
function (sara_message _msg)
  message (STATUS "[Sara] ${_msg}")
endfunction ()


function (sara_step_message _msg)
  message ("[Sara] ${_msg}")
endfunction ()


function (sara_substep_message _msg)
  message ("       ${_msg}")
endfunction ()



# ==============================================================================
# Useful macros
#
macro (sara_dissect_version)
  # Retrieve the build number.
  execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set (DO_Sara_BUILD_NUMBER "r${GIT_COMMIT_HASH}")

  # Build the version.
  set(DO_Sara_VERSION
    "${DO_Sara_VERSION_MAJOR}.${DO_Sara_VERSION_MINOR}.0") #.${DO_Sara_BUILD_NUMBER}")
  set(DO_Sara_SOVERSION "${DO_Sara_VERSION_MAJOR}.${DO_Sara_VERSION_MINOR}")

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sara_version.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/sara_version.cmake @ONLY)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sara_version.cmake.in
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sara_version.cmake @ONLY)

  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/DO/Sara/Defines.hpp.in
    ${CMAKE_BINARY_DIR}/cpp/src/DO/Sara/Defines.hpp @ONLY)

  include_directories(${CMAKE_BINARY_DIR}/cpp/src)
endmacro ()


macro (sara_append_components _component_list _component)
  set(DO_Sara_${_component}_USE_FILE UseDOSara${_component})
  list(APPEND "${_component_list}" ${_component})
endmacro ()


macro (sara_populate_available_components)
  # This macro populates the list of components and stores it in the following
  # variable:
  # - DO_Sara_COMPONENTS
  # For each component ${COMPONENT} in ${DO_Sara_COMPONENTS}, we define the
  # following variable:
  # - DO_Sara_${COMPONENT}_USE_FILE

  # Base libraries.
  sara_append_components(DO_Sara_COMPONENTS Core)

  if (NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    sara_append_components(DO_Sara_COMPONENTS Graphics)
    sara_append_components(DO_Sara_COMPONENTS FileSystem)

    # Image and Video I/O.
    sara_append_components(DO_Sara_COMPONENTS ImageIO)
    if (SARA_BUILD_VIDEOIO)
      sara_append_components(DO_Sara_COMPONENTS VideoIO)
    endif ()

    # Image processing.
    sara_append_components(DO_Sara_COMPONENTS ImageProcessing)

    # Feature detection and description.
    sara_append_components(DO_Sara_COMPONENTS Features)
    sara_append_components(DO_Sara_COMPONENTS FeatureDetectors)
    sara_append_components(DO_Sara_COMPONENTS FeatureDescriptors)

    # Feature matching.
    sara_append_components(DO_Sara_COMPONENTS Match)
    sara_append_components(DO_Sara_COMPONENTS FeatureMatching)

    # Multiple view geometry.
    sara_append_components(DO_Sara_COMPONENTS MultiViewGeometry)
    sara_append_components(DO_Sara_COMPONENTS SfM)

    # Disjoint sets.
    sara_append_components(DO_Sara_COMPONENTS DisjointSets)

    # Geometry.
    sara_append_components(DO_Sara_COMPONENTS Geometry)

    # KDTree for fast neighbor search.
    sara_append_components(DO_Sara_COMPONENTS KDTree)

    # Visualization
    sara_append_components(DO_Sara_COMPONENTS Visualization)
  endif ()

  # DEBUG: Print the list of component libraries.
  sara_step_message("Currently available components in Sara:")
  foreach (component ${DO_Sara_COMPONENTS})
    sara_substep_message ("- ${component}")
  endforeach (component)
endmacro ()


macro (sara_check_requested_components)
  # This macro defines one variable:
  # - DO_Sara_USE_COMPONENTS which lists all the compiled compiled libraries we
  #   want to use.

  sara_step_message("Requested libraries by project '${PROJECT_NAME}':")
  foreach (component ${DO_Sara_FIND_COMPONENTS})
    message ("      - ${component}")
  endforeach (component)

  set(DO_Sara_USE_COMPONENTS "")

  foreach (COMPONENT ${DO_Sara_FIND_COMPONENTS})
    # By default, mark the requested component as not found.
    set(DO_Sara_${COMPONENT}_FOUND FALSE)

    # Now check if the requested component exists.
    list(FIND DO_Sara_COMPONENTS ${COMPONENT} COMPONENT_INDEX)
    if (NOT COMPONENT_INDEX EQUAL -1)
      set(DO_Sara_${COMPONENT}_FOUND TRUE)
      list (APPEND DO_Sara_USE_COMPONENTS ${COMPONENT})
    endif ()

    # Stop if REQUIRED option was given.
    if (NOT DO_Sara_${COMPONENT}_FOUND AND DO_Sara_FIND_REQUIRED)
      message (FATAL_ERROR "[Sara] ${COMPONENT} does not exist!")
    endif ()
  endforeach ()
endmacro ()



# ==============================================================================
# Utility macros to easily add a new C++ library.
#
macro (sara_create_common_variables _library_name)
  set(DO_Sara_${_library_name}_SOURCE_DIR
    ${DO_Sara_SOURCE_DIR}/${_library_name}
    CACHE STRING "Source directory")

  if ("${DO_Sara_${_library_name}_SOURCE_FILES}" STREQUAL "")
    set(DO_Sara_${_library_name}_LIBRARIES ""
      CACHE STRING "Library name")
  else ()
    set(DO_Sara_${_library_name}_LIBRARIES
      DO_Sara_${_library_name}
      CACHE STRING "Library name")
  endif ()
endmacro ()


macro (sara_append_subdir_files _parentdir _child_dir _hdr_list_var _src_list_var)
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


macro(sara_glob_directory _curdir)
  #message(STATUS "Parsing current source directory = ${_curdir}")
  file(GLOB curdir_children RELATIVE ${_curdir} ${_curdir}/*)

  get_filename_component(curdir_name "${_curdir}" NAME)
  #message("Directory name: ${curdir_name}")

  file(GLOB DO_Sara_${curdir_name}_HEADER_FILES FILES ${_curdir}/*.hpp)
  file(GLOB DO_Sara_${curdir_name}_SOURCE_FILES FILES ${_curdir}/*.cpp)

  foreach (child ${curdir_children})
    if (IS_DIRECTORY ${_curdir}/${child} AND NOT "${child}" STREQUAL "build")
      #message("Parsing child directory = '${child}'")
      sara_append_subdir_files(${_curdir} ${child}
                             DO_Sara_${curdir_name}_HEADER_FILES
                             DO_Sara_${curdir_name}_SOURCE_FILES)
    endif ()
  endforeach ()

  set(DO_Sara_${curdir_name}_MASTER_HEADER ${DO_Sara_SOURCE_DIR}/${curdir_name}.hpp)
  source_group("Master Header File" FILES ${DO_Sara_${curdir_name}_MASTER_HEADER})

  list(APPEND DO_Sara_${curdir_name}_HEADER_FILES
       ${DO_Sara_${curdir_name}_MASTER_HEADER})

  #message(STATUS "Master Header:\n ${DO_Sara_${curdir_name}_MASTER_HEADER}")
  #message(STATUS "Header file list:\n ${DO_Sara_${curdir_name}_HEADER_FILES}")
  #message(STATUS "Source file list:\n ${DO_Sara_${curdir_name}_SOURCE_FILES}")
endmacro()


macro (sara_append_library _library_name
                           _include_dirs
                           _hdr_files
                           _src_files
                           _lib_dependencies)
  # Verbose comment.
  sara_message("[Sara] Creating project 'DO_Sara_${_library_name}'")

  # Bookmark the project to make sure the library is created only once.
  set_property(GLOBAL PROPERTY _DO_Sara_${_library_name}_INCLUDED 1)

  # Create the project:
  if (NOT "${_src_files}" STREQUAL "")
    # - Case 1: the project contains 'cpp' source files
    #   Specify the source files.
    add_library(DO_Sara_${_library_name}
      ${DO_Sara_SOURCE_DIR}/UseDOSara${_library_name}.cmake
      ${_hdr_files} ${_src_files})
    add_library(DO::Sara::${_library_name} ALIAS DO_Sara_${_library_name})

    # For every single library in Sara.
    target_include_directories(DO_Sara_${_library_name}
      PUBLIC
      $<BUILD_INTERFACE:${DO_Sara_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${DO_Sara_ThirdParty_DIR}>
      $<BUILD_INTERFACE:${DO_Sara_ThirdParty_DIR}/eigen>
      $<INSTALL_INTERFACE:include>)

    # Form the compiled library output name.
    set(_library_output_basename DO_Sara_${_library_name})
    if (SARA_BUILD_SHARED_LIBS)
      set (_library_output_name "${_library_output_basename}")
      set (_library_output_name_debug "${_library_output_basename}-d")
    else ()
      set (_library_output_name "${_library_output_basename}-s")
      set (_library_output_name_debug "${_library_output_basename}-sd")
    endif ()

    # Specify output name and version.
    set_target_properties(DO_Sara_${_library_name}
      PROPERTIES
      VERSION ${DO_Sara_VERSION}
      SOVERSION ${DO_Sara_SOVERSION}
      OUTPUT_NAME ${_library_output_name}
      OUTPUT_NAME_DEBUG ${_library_output_name_debug})

    # Set correct compile definitions when building the libraries.
    if (SARA_BUILD_SHARED_LIBS)
      target_compile_definitions(DO_Sara_${_library_name}
        PRIVATE DO_SARA_EXPORTS)
    else ()
      target_compile_definitions(DO_Sara_${_library_name}
        PUBLIC DO_SARA_STATIC)
    endif ()

    # Specify where to install the static library.
    install(
      TARGETS DO_Sara_${_library_name}
      RUNTIME DESTINATION bin COMPONENT Libraries
      LIBRARY DESTINATION lib COMPONENT Libraries
      ARCHIVE DESTINATION lib COMPONENT Libraries)
  else ()
    # - Case 2: the project is a header-only library
    #   Specify the source files.
    message(STATUS
      "[Sara] No linking needed for header-only project "
      "'DO_Sara_${_library_name}'")
    add_library(DO_Sara_${_library_name} INTERFACE)
    target_sources(DO_Sara_${_library_name}
      INTERFACE
      ${DO_Sara_DIR}/cmake/UseDOSara${_library_name}.cmake
      ${_hdr_files} ${_src_files})

    if(MSVC)
      add_custom_target(DO_Sara_${_library_name}
        SOURCES
        ${DO_Sara_DIR}/cmake/UseDOSara${_library_name}.cmake
        ${_hdr_files})
    endif()

    target_include_directories(DO_Sara_${_library_name}
      INTERFACE
      $<BUILD_INTERFACE:${DO_Sara_SOURCE_DIR}>
      $<BUILD_INTERFACE:${_include_dirs}>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  endif ()

  # Drop older compiler support in favor of C++17.
  set_target_properties(DO_Sara_${_library_name}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    FOLDER "Libraries/Sara")

  # Propagate C++17 to any project linking against the library.
  target_compile_features(DO_Sara_${_library_name}
    INTERFACE cxx_std_17)

  # Figure out the rest later.
  # 6. Specify where to install the library.
  # install(TARGETS DO-Sara-${_library_name}
  #   EXPORT DO-Sara-${_library_name}-Targets
  #   RUNTIME  DESTINATION bin     COMPONENT Libraries
  #   LIBRARY  DESTINATION lib     COMPONENT Libraries
  #   ARCHIVE  DESTINATION lib     COMPONENT Libraries
  #   INCLUDES DESTINATION include COMPONENT LIBRARIES)

  # # 7. Versioning.
  # write_basic_package_version_file(
  #   "DO-Sara-${_library_name}-Version.cmake"
  #   VERSION DO-Sara-${DO_Sara_VERSION}
  #   COMPATIBILITY SameMajorVersion)

  # install(FILES "DO-Sara-${_library_name}-Config.cmake"
  #               "DO-Sara-${_library_name}-Version.cmake"
  #         DESTINATION lib/cmake/DO)

  # 8. For library reuse.
  # install(EXPORT DO-Sara-${_library_name}-Targets
  #   FILE DO-Sara-${_library_name}-Targets.cmake
  #   NAMESPACE DO::Sara::
  #   DESTINATION lib/cmake/DO)
  # foreach (lib ${_lib_dependencies})
  #   find_dependency(${_lib} ${DO_Sara_VERSION})
  # endforeach ()
  # include("${CMAKE_CURRENT_LIST_DIR}/DO-Sara-${_library_name}-Targets.cmake")
endmacro ()


macro (sara_generate_library _library_name)
  sara_append_library(${_library_name}
    "${DO_Sara_SOURCE_DIR}"
    "${DO_Sara_${_library_name}_HEADER_FILES}"
    "${DO_Sara_${_library_name}_SOURCE_FILES}"
    "${DO_Sara_${_library_name}_LINK_LIBRARIES}")
endmacro ()



# ==============================================================================
# Specific macro to add a unit test
#
# Bookmark the list of all unit tests in a global property.
set_property(GLOBAL PROPERTY _DO_SARA_TESTS "")

function (sara_add_test)
  set(_options OPTIONS)
  set(_single_value_args FOLDER NAME)
  set(_multiple_value_args SOURCES DEPENDENCIES)
  cmake_parse_arguments(test
    "${_options}" "${_single_value_args}" "${_multiple_value_args}" ${ARGN})


  if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    xctest_add_bundle(${test_NAME} DO_Sara_Core
      ${test_SOURCES}
      /Users/david/GitLab/DO-CV/sara/cpp/test/BoostToXCTest/BoostToXCTest.mm)
    target_link_libraries(${test_NAME}
      PRIVATE
      Boost::system
      Boost::unit_test_framework
      ${test_DEPENDENCIES})
    target_compile_definitions(${test_NAME}
      PRIVATE
      BOOST_ALL_NO_LIB
      BOOST_TEST_NO_MAIN
      BOOST_TEST_ALTERNATIVE_INIT_API)
    xctest_add_test(XCTest.${test_NAME} ${test_NAME})
  else ()
    # Create the unit test project.
    add_executable(${test_NAME} ${test_SOURCES})
    target_include_directories(${test_NAME}
      PRIVATE
      ${Boost_INCLUDE_DIR})
    target_link_libraries(${test_NAME}
      PRIVATE
      ${Boost_LIBRARIES}
      ${test_DEPENDENCIES})
    target_compile_definitions(${test_NAME}
      PRIVATE
      BOOST_ALL_NO_LIB
      BOOST_TEST_DYN_LINK)

    set_target_properties(${test_NAME}
      PROPERTIES
      COMPILE_FLAGS ${SARA_DEFINITIONS}
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    add_test(NAME ${test_NAME} COMMAND $<TARGET_FILE:${test_NAME}>)
  endif ()

  if (test_FOLDER)
    set_property(
      TARGET ${test_NAME}
      PROPERTY FOLDER "Tests/Sara/${test_FOLDER}")
  endif ()

  get_property(DO_SARA_TESTS GLOBAL PROPERTY _DO_SARA_TESTS)
  list(APPEND DO_SARA_TESTS ${test_NAME})
  set_property(GLOBAL PROPERTY _DO_SARA_TESTS "${DO_SARA_TESTS}")
endfunction ()
