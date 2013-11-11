macro(do_get_os_info)
  string(REGEX MATCH "Linux" OS_IS_LINUX ${CMAKE_SYSTEM_NAME})
  set(DO_LIB_INSTALL_DIR "lib")
  set(DO_INCLUDE_INSTALL_DIR
      "include/DO-${DO_MAJOR_VERSION}.${DO_MINOR_VERSION}")
endmacro(do_get_os_info)

macro(do_dissect_version)
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
endmacro(do_dissect_version)




################################################################################
# Helper macros
# 
macro (do_message MSG)
  message (STATUS "[DO] ${MSG}")
endmacro (do_message MSG)

macro (do_step_message MSG)
  message ("[DO] ${MSG}")
endmacro (do_step_message MSG)

macro (do_substep_message MSG)
  message ("     ${MSG}")
endmacro (do_substep_message MSG)


macro (do_append_components COMPONENTLIST COMPONENT)
  set(DO_${COMPONENT}_LIBRARIES DO_${COMPONENT})
  set(DO_${COMPONENT}_USE_FILE UseDO${COMPONENT})
  list(APPEND "${COMPONENTLIST}" ${COMPONENT})
endmacro (do_append_components)


macro (do_list_files SOURCE_FILES REL_PATH EXTENSION)
  file(GLOB SOURCE_FILES
       RELATIVE ${REL_PATH}
       FILES_MATCHING PATTERN ${EXTENSION})
  foreach (l ${LIST})
    set(l ${PATH}/l)
    message (l)
  endforeach ()
  message (${LIST})
endmacro (do_list_files)


macro (do_append_library NAME            # Library name
                         LIBRARY_TYPE    # shared or static
                         INCLUDE_DIRS    # include directories
                         HEADER_FILES    # header files needed to build library
                         SOURCE_FILES    # source files needed to build library
                         LINK_LIBRARIES) # library dependencies
  get_property(DO_${NAME}_ADDED GLOBAL PROPERTY _DO_${NAME}_INCLUDED)
  if (NOT DO_${NAME}_ADDED)
    # 1. Verbose
    #message(STATUS "[DO] Creating project 'DO${NAME}'")
    # 2. Bookmark the project to make sure we don't try to add the library 
    #    more than once.
    set_property(GLOBAL PROPERTY _DO_${NAME}_INCLUDED 1)
    # 3. Include third-party library directories.
    if (NOT "${INCLUDE_DIRS}" STREQUAL "")
      include_directories(${INCLUDE_DIRS})
    endif ()
    # 4. Create the project:
    if (NOT "${SOURCE_FILES}" STREQUAL "")
      # - Case 1: the project contains 'cpp' source files
      #   Specify the source files.
      add_library(DO_${NAME} ${LIBRARY_TYPE} ${HEADER_FILES} ${SOURCE_FILES})
      # Link with external libraries
      #message(STATUS 
      #        "[DO] Linking project 'DO${NAME}' with '${LINK_LIBRARIES}'")
      target_link_libraries(DO_${NAME} ${LINK_LIBRARIES})
    else ()
      # - Case 2: the project is a header-only library
      #   Specify the source files.
      #add_library(DO_${NAME} STATIC ${HEADER_FILES})
      #message(STATUS 
      #        "[DO] No linking needed for header-only project 'DO.${NAME}'")
      #set_target_properties(DO_${NAME} PROPERTIES LINKER_LANGUAGE CXX)
      add_custom_target(DO_${NAME} SOURCES ${HEADER_FILES})
    endif ()
  endif ()
  set_property(TARGET DO_${NAME} PROPERTY FOLDER "DO Modules")
endmacro (do_append_library)


macro (do_unit_test NAME SOURCES EXTRA_LIBS)
  include_directories(${gtest_DIR}/include)
  add_executable(DO_${NAME}_test ${SOURCES})
  target_link_libraries(DO_${NAME}_test
                        ${EXTRA_LIBS} # Extra libs MUST be first.
                        gtest)
  set_target_properties(DO_${NAME}_test PROPERTIES
                        COMPILE_FLAGS -DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}
                        COMPILE_DEFINITIONS DO_STATIC)
  add_test(DO_${NAME}_test "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/DO_${NAME}_test")
endmacro (do_unit_test)


macro (do_set_specific_target_properties TARGET COMPILE_DEFS)
  set_target_properties(${TARGET} PROPERTIES
                        VERSION ${DO_VERSION}
                        SOVERSION ${DO_SOVERSION}
                        COMPILE_DEFINITIONS ${COMPILE_DEFS})
  if (WIN32)
    set_target_properties(${TARGET} PROPERTIES
                          OUTPUT_NAME_DEBUG   ${TARGET}-${DO_VERSION}-debug
                          OUTPUT_NAME_RELEASE ${TARGET}-${DO_VERSION}-release)
  endif ()
endmacro (do_set_specific_target_properties)
