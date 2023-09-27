function(sara_add_halide_library TARGET)
   ##
   # Set up argument parsing for extra outputs.
   ##

   # See Module.cpp for list of extra outputs. The following outputs intentionally do not appear:
   # - `c_header` is always generated
   # - `c_source` is selected by C_BACKEND
   # - `object` is selected for CMake-target-compile
   # - `static_library` is selected for cross-compile
   # - `cpp_stub` is not available
   set(extra_output_names
       ASSEMBLY
       BITCODE
       COMPILER_LOG
       FEATURIZATION
       LLVM_ASSEMBLY
       PYTHON_EXTENSION
       PYTORCH_WRAPPER
       REGISTRATION
       SCHEDULE
       STMT
       STMT_HTML)

   # "hash table" of extra outputs to extensions
   set(ASSEMBLY_extension ".s")
   set(BITCODE_extension ".bc")
   set(COMPILER_LOG_extension ".halide_compiler_log")
   set(FEATURIZATION_extension ".featurization")
   set(LLVM_ASSEMBLY_extension ".ll")
   set(PYTHON_EXTENSION_extension ".py.cpp")
   set(PYTORCH_WRAPPER_extension ".pytorch.h")
   set(REGISTRATION_extension ".registration.cpp")
   set(SCHEDULE_extension ".schedule.h")
   set(STMT_extension ".stmt")
   set(STMT_HTML_extension ".stmt.html")

   ##
   # Parse the arguments and set defaults for missing values.
   ##

   set(options C_BACKEND GRADIENT_DESCENT)
   set(oneValueArgs FROM GENERATOR FUNCTION_NAME NAMESPACE USE_RUNTIME AUTOSCHEDULER HEADER ${extra_output_names})
   set(multiValueArgs TARGETS FEATURES PARAMS PLUGINS)
   cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

   if (NOT "${ARG_UNPARSED_ARGUMENTS}" STREQUAL "")
       message(AUTHOR_WARNING "Arguments to add_halide_library were not recognized: ${ARG_UNPARSED_ARGUMENTS}")
   endif ()

   if (NOT ARG_FROM)
       message(FATAL_ERROR "Missing FROM argument specifying a Halide generator target")
   endif ()

   _Halide_place_dll(${ARG_FROM})

   if (ARG_C_BACKEND)
       if (ARG_USE_RUNTIME)
           message(AUTHOR_WARNING "The C backend does not use a runtime.")
       endif ()
       if (ARG_TARGETS)
           message(AUTHOR_WARNING "The C backend sources will be compiled with the current CMake toolchain.")
       endif ()
   endif ()

   set(gradient_descent "$<BOOL:${ARG_GRADIENT_DESCENT}>")

   if (NOT ARG_GENERATOR)
       set(ARG_GENERATOR "${TARGET}")
   endif ()

   if (NOT ARG_FUNCTION_NAME)
       set(ARG_FUNCTION_NAME "${TARGET}")
   endif ()

   if (ARG_NAMESPACE)
       set(ARG_FUNCTION_NAME "${ARG_NAMESPACE}::${ARG_FUNCTION_NAME}")
   endif ()

   # If no TARGETS argument, use Halide_TARGET instead
   if (NOT ARG_TARGETS)
       set(ARG_TARGETS "${Halide_TARGET}")
   endif ()

   # If still no TARGET, try to use host, but if that would
   # cross-compile, then default to 'cmake' and warn.
   if (NOT ARG_TARGETS)
       if (Halide_HOST_TARGET STREQUAL Halide_CMAKE_TARGET)
           set(ARG_TARGETS host)
       else ()
           message(AUTHOR_WARNING
                   "Targets must be manually specified to add_halide_library when cross-compiling. "
                   "The default 'host' target ${Halide_HOST_TARGET} differs from the active CMake "
                   "target ${Halide_CMAKE_TARGET}. Using ${Halide_CMAKE_TARGET} to compile ${TARGET}. "
                   "This might result in performance degradation from missing arch flags (eg. avx).")
           set(ARG_TARGETS "${Halide_CMAKE_TARGET}")
       endif ()
   endif ()

   list(TRANSFORM ARG_TARGETS REPLACE "cmake" "${Halide_CMAKE_TARGET}")

   list(APPEND ARG_FEATURES no_runtime)
   list(JOIN ARG_FEATURES "-" ARG_FEATURES)
   list(TRANSFORM ARG_TARGETS APPEND "-${ARG_FEATURES}")

   ##
   # Set up the runtime library, if needed
   ##

   if (ARG_C_BACKEND)
       # The C backend does not provide a runtime, so just supply headers.
       set(ARG_USE_RUNTIME Halide::Runtime)
   elseif (NOT ARG_USE_RUNTIME)
       # If we're not using an existing runtime, create one.
       _Halide_add_halide_runtime("${TARGET}.runtime" FROM ${ARG_FROM}
                                  TARGETS ${ARG_TARGETS})
       set(ARG_USE_RUNTIME "${TARGET}.runtime")
   elseif (NOT TARGET ${ARG_USE_RUNTIME})
       message(FATAL_ERROR "Invalid runtime target ${ARG_USE_RUNTIME}")
   else ()
       _Halide_add_targets_to_runtime(${ARG_USE_RUNTIME} TARGETS ${ARG_TARGETS})
   endif ()

   ##
   # Determine which outputs the generator call will emit.
   ##

   _Halide_get_platform_details(
           is_crosscompiling
           object_suffix
           static_library_suffix
           ${ARG_TARGETS})

   # Always emit a C header
   set(generator_outputs c_header)
   set(generator_output_files "${TARGET}.h")
   if (ARG_HEADER)
       set(${ARG_HEADER} "${TARGET}.h" PARENT_SCOPE)
   endif ()

   # Then either a C source, a set of object files, or a cross-compiled static library.
   if (ARG_C_BACKEND)
       list(APPEND generator_outputs c_source)
       set(generator_sources "${TARGET}.halide_generated.cpp")
   elseif (is_crosscompiling)
       # When cross-compiling, we need to use a static, imported library
       list(APPEND generator_outputs static_library)
       set(generator_sources "${TARGET}${static_library_suffix}")
   else ()
       # When compiling for the current CMake toolchain, create a native
       list(APPEND generator_outputs object)
       list(LENGTH ARG_TARGETS len)
       if (len EQUAL 1)
           set(generator_sources "${TARGET}${object_suffix}")
       else ()
           set(generator_sources ${ARG_TARGETS})
           list(TRANSFORM generator_sources PREPEND "${TARGET}-")
           list(TRANSFORM generator_sources APPEND "${object_suffix}")
           list(APPEND generator_sources "${TARGET}_wrapper${object_suffix}")
       endif ()
   endif ()
   list(APPEND generator_output_files ${generator_sources})

   # Add in extra outputs using the table defined at the start of this function
   foreach (out IN LISTS extra_output_names)
       if (ARG_${out})
           set(${ARG_${out}} "${TARGET}${${out}_extension}" PARENT_SCOPE)
           list(APPEND generator_output_files "${TARGET}${${out}_extension}")
           string(TOLOWER "${out}" out)
           list(APPEND generator_outputs ${out})
       endif ()
   endforeach ()

   ##
   # Attach an autoscheduler if the user requested it
   ##

   set(autoscheduler "")
   if (ARG_AUTOSCHEDULER)
       if ("${ARG_AUTOSCHEDULER}" MATCHES "::")
           if (NOT TARGET "${ARG_AUTOSCHEDULER}")
               message(FATAL_ERROR "Autoscheduler ${ARG_AUTOSCHEDULER} does not exist.")
           endif ()

           # Convention: if the argument names a target like "Namespace::Scheduler" then
           # it is assumed to be a MODULE target providing a scheduler named "Scheduler".
           list(APPEND ARG_PLUGINS "${ARG_AUTOSCHEDULER}")
           string(REGEX REPLACE ".*::(.*)" "\\1" ARG_AUTOSCHEDULER "${ARG_AUTOSCHEDULER}")
       elseif (NOT ARG_PLUGINS)
           message(AUTHOR_WARNING "AUTOSCHEDULER set to a scheduler name but no plugins were loaded")
       endif ()
       set(autoscheduler -s "${ARG_AUTOSCHEDULER}")
       list(PREPEND ARG_PARAMS auto_schedule=true)
   endif ()

   ##
   # Main library target for filter.
   ##

   if (is_crosscompiling)
       add_library("${TARGET}" STATIC IMPORTED GLOBAL)
       set_target_properties("${TARGET}" PROPERTIES
                             IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${generator_sources}")
   else ()
       # THE CORRECT CODE IS AS FOLLOWS:
       list(APPEND TARGET_OBJECT_FILES
           ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}${object_suffix}
           ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}.runtime${object_suffix})

       add_library("${TARGET}" OBJECT IMPORTED GLOBAL)
       set_target_properties("${TARGET}" PROPERTIES
           IMPORTED_OBJECTS "${TARGET_OBJECT_FILES}"
           POSITION_INDEPENDENT_CODE ON
           LINKER_LANGUAGE CXX)

       # THE ORIGINAL CODE
       # add_library("${TARGET}" STATIC ${generator_sources})
       # set_target_properties("${TARGET}" PROPERTIES
       #                       POSITION_INDEPENDENT_CODE ON
       #                       LINKER_LANGUAGE CXX)
   endif ()

   # Load the plugins and setup dependencies
   set(generator_plugins "")
   if (ARG_PLUGINS)
       foreach (p IN LISTS ARG_PLUGINS)
           list(APPEND generator_plugins "$<TARGET_FILE:${p}>")
       endforeach ()
       set(generator_plugins -p "$<JOIN:${generator_plugins},$<COMMA>>")
   endif ()

   add_custom_command(OUTPUT ${generator_output_files}
                      COMMAND ${ARG_FROM}
                      -n "${TARGET}"
                      -d "${gradient_descent}"
                      -g "${ARG_GENERATOR}"
                      -f "${ARG_FUNCTION_NAME}"
                      -e "$<JOIN:${generator_outputs},$<COMMA>>"
                      ${generator_plugins}
                      ${autoscheduler}
                      -o .
                      "target=$<JOIN:${ARG_TARGETS},$<COMMA>>"
                      ${ARG_PARAMS}
                      DEPENDS "${ARG_FROM}" ${ARG_PLUGINS}
                      VERBATIM)

   list(TRANSFORM generator_output_files PREPEND "${CMAKE_CURRENT_BINARY_DIR}/")
   add_custom_target("${TARGET}.update" ALL DEPENDS ${generator_output_files})

   add_dependencies("${TARGET}" "${TARGET}.update")

   target_include_directories("${TARGET}" INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

   # NOT NEEDED BECAUSE OF OUR WORKAROUND.
   # target_link_libraries("${TARGET}" INTERFACE "${ARG_USE_RUNTIME}")
endfunction()