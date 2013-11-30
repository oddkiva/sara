# Author: Davorin Učakar <davorin.ucakar@gmail.com>

# Example:
#
#   add_pch( pch stable.h stable.c )
#   add_executable( test test.c )
#   use_pch( test pch )
#
# where: - `pch` is PCH target name
#        - `stable.h` is a header that should be precompiled.
#        - `stable.c` is a dummy module that contains only an include directive for `stable.h` (it
#          is required for proper dependency resolution to trigger recompilation of PCH).
# Notes: - Only works for GCC and LLVM/Clang.
#        - Compiler flags are retrieved from `CMAKE_CXX_FLAGS`, `CMAKE_CXX_FLAGS_<BUILDTYPE>`,
#          included directories added via `include_directories()` and defines added via
#          `add_definitions()`.

include( CheckCXXSourceCompiles )

check_cxx_source_compiles( "
#ifdef __clang__
int main( int, char** ) { return 0; }
#else
# error Not LLVM/Clang
#endif
" CLANG )

macro( add_pch _pchTarget _inputHeader _inputModule )
  # Extract CMAKE_CXX_FLAGS and CMAKE_CXX_FLAGS_XXX for the current configuration XXX.
  string( TOUPPER "CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}" _build_type_flags_var )
  set( _flags "${CMAKE_CXX_FLAGS} ${${_build_type_flags_var}}" )

  # Convert string of space separated flags into a list.
  separate_arguments( _flags )

  # Extract include directories set by include_directories command.
  get_directory_property( _includes INCLUDE_DIRECTORIES )
  foreach( _include ${_includes} )
    list( APPEND _flags "-I${_include}" )
  endforeach()

  # Extract definitions set by add_definitions command.
  get_directory_property( _defines COMPILE_DEFINITIONS )
  foreach( _define ${_defines} )
    list( APPEND _flags "-D${_define}" )
  endforeach()

  # Helper target that properly triggers recompilation of precompiled header.
  add_library( ${_pchTarget}_trigger STATIC "${_inputModule}" )

  # Build precompiled header and copy original header to the build folder - GCC wants it there.
  add_custom_command( OUTPUT "${_inputHeader}.gch"
    DEPENDS ${_pchTarget}_trigger
    COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/${_inputHeader}" "${_inputHeader}"
    COMMAND "${CMAKE_COMMAND}" -E remove -f "${_inputHeader}.gch"
    COMMAND "${CMAKE_CXX_COMPILER}" ${_flags} -o "${_inputHeader}.gch" "${CMAKE_CURRENT_SOURCE_DIR}/${_inputHeader}" )
  add_custom_target( ${_pchTarget} DEPENDS "${_inputHeader}.gch" )

  set( ${_pchTarget}_outputPCH "${CMAKE_CURRENT_BINARY_DIR}/${_inputHeader}.gch" )
endmacro()

macro( use_pch _target _pchTarget )
  add_dependencies( ${_target} ${_pchTarget} )

  # Additional parameters for LLVM/Clang.
  if( CLANG )
    set_target_properties( ${_target} PROPERTIES COMPILE_FLAGS "-include-pch ${${_pchTarget}_outputPCH}" )
  endif()
endmacro()
