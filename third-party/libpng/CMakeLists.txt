# Copyright (C) 2007-2012 LuaDist.
# Modified by David Manura.
# Redistribution and use of this file is allowed according to the terms of the MIT license.
# For details see the COPYRIGHT file distributed with LuaDist.
# Please note that the package source code is licensed under its own license.
#
# Note: code is based on the original CMakeLists.txt in libpng, having its own copyright:
# Copyright (C) 2007-2011 Glenn Randers-Pehrson
# This code is released under the libpng license.
# For conditions of distribution and use, see the disclaimer
# and license in png.h

project ( libpng C )
cmake_minimum_required ( VERSION 2.8 )
include ( cmake/dist.cmake )

set ( PNGLIB_MAJOR 1 )
set ( PNGLIB_MINOR 5 )
set ( PNGLIB_RELEASE 10 )
set ( PNGLIB_NAME libpng${PNGLIB_MAJOR}${PNGLIB_MINOR} )
set ( PNGLIB_VERSION ${PNGLIB_MAJOR}.${PNGLIB_MINOR}.${PNGLIB_RELEASE} )

# needed packages
find_package ( ZLIB REQUIRED )
include_directories ( ${ZLIB_INCLUDE_DIR} )

if ( NOT WIN32 )
  find_library ( M_LIBRARY NAMES m PATHS /usr/lib /usr/local/lib )
  if ( NOT M_LIBRARY )
    message ( STATUS "math library 'libm' not found - floating point support disabled" )
  endif ( )
else ( )
  set ( M_LIBRARY "" )
  # not needed on windows
endif ( )

# COMMAND LINE OPTIONS
option ( PNG_SHARED "Build shared lib" ${BUILD_SHARED_LIB} )
option ( PNG_TESTS "Build libpng tests" YES )
option ( PNG_DEBUG "Build with debug output" NO )
option ( PNGARG "Disable ANSI-C prototypes" NO )

# SET LIBNAME
set ( PNG_LIB_NAME png )
#removed: ${PNGLIB_MAJOR}${PNGLIB_MINOR}

# Use the prebuilt pnglibconf.h file from the scripts folder
#~# TODO: fix this by building with awk; without this no cmake build can be
#~# configured directly (to do so indirectly use your local awk to build a
#~# pnglibconf.h in the build directory.)
configure_file ( ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pnglibconf.h.prebuilt ${CMAKE_CURRENT_BINARY_DIR}/pnglibconf.h )
include_directories ( ${CMAKE_CURRENT_BINARY_DIR} )

# OUR SOURCES
set ( libpng_public_hdrs png.h pngconf.h ${CMAKE_CURRENT_BINARY_DIR}/pnglibconf.h )
set ( libpng_sources ${libpng_public_hdrs} pngdebug.h pnginfo.h pngpriv.h pngstruct.h 
  png.c pngerror.c pngget.c pngmem.c pngpread.c pngread.c pngrio.c pngrtran.c pngrutil.c 
  pngset.c pngtrans.c pngwio.c pngwrite.c pngwtran.c pngwutil.c )
set ( pngtest_sources pngtest.c )
set ( pngvalid_sources contrib/libtests/pngvalid.c )

add_definitions ( -DPNG_CONFIGURE_LIBPNG )
if ( PNG_DEBUG )
  add_definitions ( -DPNG_DEBUG )
endif ( )
include_directories ( ${CMAKE_CURRENT_SOURCE_DIR} ${ZLIB_INCLUDE_DIR} )

add_library ( ${PNG_LIB_NAME} ${libpng_sources} )
target_link_libraries ( ${PNG_LIB_NAME} ${ZLIB_LIBRARY} ${M_LIBRARY} )

if ( PNG_SHARED AND WIN32 )
  set_target_properties ( ${PNG_LIB_NAME} PROPERTIES DEFINE_SYMBOL PNG_BUILD_DLL )
endif ( )

enable_testing ( )
if ( PNG_TESTS )
  # removed: AND PNG_SHARED
  # does not work with msvc due to png_lib_ver issue
  add_executable ( pngtest ${pngtest_sources} )
  target_link_libraries ( pngtest ${PNG_LIB_NAME} )
  add_test ( pngtest ./pngtest ${CMAKE_CURRENT_SOURCE_DIR}/pngtest.png )
  #
  add_executable ( pngvalid ${pngvalid_sources} )
  target_link_libraries ( pngvalid ${PNG_LIB_NAME} )
  add_test ( pngvalid ./pngvalid )
endif ( )

install_library ( ${PNG_LIB_NAME} )
install_header ( ${libpng_public_hdrs} )
#removed: install_header(${libpng_public_hdrs} INTO ${PNGLIB_NAME})
install_doc ( libpng.3 libpng.3 png.5 )

#~ removed from original: pkgconfig
#~ removed from original: create_symlink
#~ removed from original: libpng-config
#~ removed from original: man pages
#~ removed from original: install(EXPORT)
#~ removed from original: version sovesion
