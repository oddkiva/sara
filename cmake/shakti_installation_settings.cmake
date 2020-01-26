# Set output directories.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})


# List all available components for installation.
set(CPACK_COMPONENTS_ALL Sources Libraries)


if (WIN32)
  set(CPACK_PACKAGE_NAME "DO-Shakti")
else()
  set(CPACK_PACKAGE_NAME "libDO-Shakti")
endif ()
if (SHAKTI_BUILD_SHARED_LIBS)
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-shared")
else ()
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-static")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-dbg")
endif ()

set(CPACK_PACKAGE_VENDOR "DO-CV")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "DO-Shakti: C++11/CUDA-accelerated Computer Vision")
set(CPACK_RESOURCE_FILE_LICENSE "${DO_Shakti_DIR}/COPYING.README")
set(CPACK_PACKAGE_CONTACT "David OK <david.ok8@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

set(CPACK_PACKAGE_VERSION_MAJOR ${DO_Shakti_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${DO_Shakti_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${DO_Shakti_BUILD_NUMBER})
set(CPACK_PACKAGE_VERSION ${DO_Shakti_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO-Shakti")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO-Shakti-${CMAKE_BUILD_TYPE}")
endif ()



# ============================================================================ #
# Special configuration for Debian packages.
#
set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "cmake")


# ============================================================================ #
# Special settings for auto-generated SPEC file for RPM packaging.
set(CPACK_RPM_PACKAGE_RELEASE ${DO_Shakti_VERSION})
set(CPACK_RPM_PACKAGE_LICENSE "MPL v2")
set(CPACK_RPM_PACKAGE_GROUP "Applications/Multimedia")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "from distutils.sysconfig import get_python_lib; print get_python_lib()"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

list(APPEND CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
  /usr
  /usr/include
  /usr/lib
  /usr/share
  /usr/local
  /usr/local/include
  /usr/local/lib
  /usr/local/share
  ${CMAKE_INSTALL_PREFIX}
  ${CMAKE_INSTALL_PREFIX}/include
  ${CMAKE_INSTALL_PREFIX}/lib
  ${CMAKE_INSTALL_PREFIX}/lib/python2.7/site-packages/do
  ${CMAKE_INSTALL_PREFIX}/share
  $ENV{WORKON_HOME}
  $ENV{VIRTUAL_ENV}
  $ENV{VIRTUAL_ENV}/lib
  $ENV{VIRTUAL_ENV}/lib/python2
  $ENV{VIRTUAL_ENV}/lib/python2.7
  ${PYTHON_SITE_PACKAGES_DIR}
  ${PYTHON_SITE_PACKAGES_DIR}/do)

# We don't want CPackRPM to add CUDA as a required libraries.
# See:
# http://stackoverflow.com/questions/14658034/how-do-you-make-it-so-that-cpack-doesnt-add-required-libraries-to-an-rpm
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")


# ============================================================================ #
# Special configuration for Windows installer using NSIS.
#
# Installers for 32- vs. 64-bit CMake:
#  - Root install directory (displayed to end user at installer-run time)
#  - "NSIS package/display name" (text used in the installer GUI)
#  - Registry key used to store info about the installation
if(CMAKE_CL_64)
  set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
  set(CPACK_NSIS_PACKAGE_NAME
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win64")
  set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win64")
else()
  set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
  set(CPACK_NSIS_PACKAGE_NAME
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win32")
  set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY
      "${CPACK_PACKAGE_NAME} ${CPACK_PACKAGE_VERSION} Win32")
endif()
set(CPACK_NSIS_COMPRESSOR "/SOLID lzma")

set(CPACK_NSIS_DISPLAY_NAME ${CPACK_NSIS_PACKAGE_NAME})

# ============================================================================ #
# Select package generator.
if (WIN32)
  set(CPACK_GENERATOR NSIS)
elseif (UNIX)
  if (EXISTS /etc/debian_version)
    set(CPACK_GENERATOR "DEB")
  else ()
    set(CPACK_GENERATOR "RPM")
  endif ()
endif()
