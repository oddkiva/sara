# Where to output the binaries.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Install the sources.
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  # Eigen 3
  do_message("Installing Eigen")
  install(DIRECTORY ${DO_Sara_ThirdParty_DIR}/eigen/Eigen
          DESTINATION include
          COMPONENT ThirdParty)
  set(CPACK_COMPONENT_ThirdParty_REQUIRED 1)

  # DO-${DO_PROJECT_NAME} source files
  install(FILES ${DO_${DO_PROJECT_NAME}_DIR}/COPYING.README
                ${DO_${DO_PROJECT_NAME}_DIR}/COPYING.MPL2
          DESTINATION include/DO/${DO_PROJECT_NAME}
          COMPONENT Sources)
  install(DIRECTORY ${DO_${DO_PROJECT_NAME}_DIR}/cmake
          DESTINATION include/DO/${DO_PROJECT_NAME}
          COMPONENT Sources)
  install(DIRECTORY ${DO_${DO_PROJECT_NAME}_DIR}/src/DO
          DESTINATION include/
          COMPONENT Sources)
  install(FILES ${DO_${DO_PROJECT_NAME}_SOURCE_DIR}/Defines.hpp
          DESTINATION include/DO/${DO_PROJECT_NAME}
          COMPONENT Sources)
  set(CPACK_COMPONENT_Sources_REQUIRED 1)
endif ()

# DO-${DO_PROJECT_NAME} component libraries
foreach (component ${DO_${DO_PROJECT_NAME}_COMPONENTS})
  do_message("Installing DO_${DO_PROJECT_NAME}_${component}")
  include(${DO_${DO_PROJECT_NAME}_${component}_USE_FILE})
endforeach (component)


# List all available components for installation.
set(CPACK_COMPONENTS_ALL ThirdParty Sources Libraries)


if (WIN32)
  set(CPACK_PACKAGE_NAME "DO-Sara")
else()
  set(CPACK_PACKAGE_NAME "libDO-Sara")
endif ()
if (DO_BUILD_SHARED_LIBS)
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-shared")
else ()
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-static")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME}-dbg")
endif ()

set(CPACK_PACKAGE_VENDOR "DO-CV")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "DO-Sara: An easy-to-use C++ set of libraries for computer vision")
set(CPACK_RESOURCE_FILE_LICENSE "${DO_${DO_PROJECT_NAME}_DIR}/COPYING.README")
set(CPACK_PACKAGE_CONTACT "David OK")

set(CPACK_PACKAGE_VERSION_MAJOR ${DO_${DO_PROJECT_NAME}_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${DO_${DO_PROJECT_NAME}_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${DO_${DO_PROJECT_NAME}_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${DO_${DO_PROJECT_NAME}_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO-${DO_PROJECT_NAME}")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "DO-${DO_PROJECT_NAME}-${CMAKE_BUILD_TYPE}")
endif ()



# ============================================================================ #
# Special configuration for Debian packages.
#
set(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
#set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_DEPENDS
    "cmake, libjpeg-dev, libpng-dev, libtiff5-dev, qtbase5-dev")



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
  set(CPACK_GENERATOR "DEB")
endif()
